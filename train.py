import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial
from prettytable import PrettyTable

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
# import wandb
from utils.dataset import RefDataset, EndoVisDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    cfgs = get_parser()
    cfgs.manual_seed = init_random_seed(cfgs.manual_seed)
    set_random_seed(cfgs.manual_seed, deterministic=False)

    cfgs.ngpus_per_node = torch.cuda.device_count()
    cfgs.world_size = cfgs.ngpus_per_node * cfgs.world_size
    if cfgs.world_size == 1:
        main_worker(0, cfgs)
    else:
        mp.spawn(main_worker, nprocs=cfgs.ngpus_per_node, args=(cfgs, ))


def main_worker(gpu, cfgs):
    cfgs.output_dir = os.path.join(cfgs.output_folder, cfgs.exp_name)

    # local rank & global rank
    cfgs.gpu = gpu
    cfgs.rank = cfgs.rank * cfgs.ngpus_per_node + gpu
    torch.cuda.set_device(cfgs.gpu)

    # logger
    setup_logger(cfgs.output_dir,
                 distributed_rank=cfgs.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=cfgs.dist_backend,
                            init_method=cfgs.dist_url,
                            world_size=cfgs.world_size,
                            rank=cfgs.rank)

    # wandb
    # if cfgs.rank == 0:
    #     wandb.init(job_type="training",
    #                mode="online",
    #                config=cfgs,
    #                project="CRIS",
    #                name=cfgs.exp_name,
    #                tags=[cfgs.dataset, cfgs.clip_pretrain])
    dist.barrier()

    # build model
    model, param_list = build_segmenter(cfgs)
    if cfgs.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # freeze
    for name in cfgs.freeze_modules:
        for n, p in model.named_parameters():
            if n.startswith(name) or n.startswith('module.{}'.format(name)):
                p.requires_grad = False
    logger.info(model)
    table = PrettyTable(['Name', 'Shape', 'ReqGrad'])
    for n, p in model.named_parameters():
        table.add_row([n, p.shape, p.requires_grad])
    table.align = 'l'
    logger.info('\n' + table.get_string())
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[cfgs.gpu],
                                                find_unused_parameters=True)

    # build optimizer & lr scheduler
    if cfgs.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list,
                                     lr=cfgs.base_lr,
                                     weight_decay=cfgs.weight_decay,
                                     amsgrad=cfgs.amsgrad)
    elif cfgs.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list,
                                      lr=cfgs.base_lr,
                                      weight_decay=cfgs.weight_decay,
                                      amsgrad=cfgs.amsgrad)
    else:
        assert 'Not support optimizer: {}!'.format(cfgs.optimizer)
    scheduler = MultiStepLR(optimizer,
                            milestones=cfgs.milestones,
                            gamma=cfgs.lr_decay)
    scaler = amp.GradScaler()

    # build dataset
    cfgs.batch_size = int(cfgs.batch_size / cfgs.ngpus_per_node)
    cfgs.batch_size_val = int(cfgs.batch_size_val / cfgs.ngpus_per_node)
    cfgs.workers = int(
        (cfgs.workers + cfgs.ngpus_per_node - 1) / cfgs.ngpus_per_node)
    train_data = EndoVisDataset(cfgs, mode='train')
    val_data = EndoVisDataset(cfgs, mode='val')

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=cfgs.workers,
                      rank=cfgs.rank,
                      seed=cfgs.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=cfgs.batch_size,
                                   shuffle=False,
                                   num_workers=cfgs.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data,
                                 batch_size=cfgs.batch_size_val,
                                 shuffle=False,
                                 num_workers=cfgs.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)

    best_IoU = 0.0
    # resume
    if os.path.exists(os.path.join(cfgs.output_dir, 'last_model.pth')):
        cfgs.resume = os.path.join(cfgs.output_dir, 'last_model.pth')
    if cfgs.resume:
        if os.path.isfile(cfgs.resume):
            logger.info("=> loading checkpoint '{}'".format(cfgs.resume))
            checkpoint = torch.load(cfgs.resume)
            cfgs.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfgs.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check cfgs.resume again!"
                .format(cfgs.resume))
        torch.cuda.empty_cache()

    # start training
    start_time = time.time()
    for epoch in range(cfgs.start_epoch, cfgs.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log,
              cfgs)

        # evaluation
        iou, prec_dict = validate(val_loader, model, epoch_log, cfgs)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(cfgs.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'prec': prec_dict,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(cfgs.output_dir, "best_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
