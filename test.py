import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import RefDataset, EndoVisDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--only_pred_first_sent', action='store_true')
    parser.add_argument('--test_sents_type',
                        default='use_class_name_sent',
                        type=str)
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    # add args to config.
    cfg.__setattr__('only_pred_first_sent', args.only_pred_first_sent)
    cfg.__setattr__('test_sents_type', args.test_sents_type)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    cfgs = get_parser()
    cfgs.output_dir = os.path.join(cfgs.output_folder, cfgs.exp_name)
    if cfgs.visualize:
        cfgs.score_dir = os.path.join(cfgs.output_dir, "score")
        if os.path.exists(cfgs.score_dir):
            os.system("rm -rf {}".format(cfgs.score_dir))
            print("remove {}!".format(cfgs.score_dir))
        os.makedirs(cfgs.score_dir, exist_ok=True)
        cfgs.vis_dir = os.path.join(cfgs.output_dir, "test_vis")
        os.makedirs(cfgs.vis_dir, exist_ok=True)
        cfgs.mae_vis_dir = os.path.join(cfgs.output_dir, "mae_vis")
        os.makedirs(cfgs.mae_vis_dir, exist_ok=True)

    # logger
    setup_logger(cfgs.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(cfgs)

    # build dataset & dataloader
    test_data = EndoVisDataset(cfgs, mode='test')
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    # build model
    model, _ = build_segmenter(cfgs)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    cfgs.model_dir = os.path.join(cfgs.output_dir, "best_model.pth")
    if os.path.isfile(cfgs.model_dir):
        logger.info("=> loading checkpoint '{}'".format(cfgs.model_dir))
        checkpoint = torch.load(cfgs.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(cfgs.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check cfgs.resume again!"
            .format(cfgs.model_dir))

    # inference
    inference(test_loader, model, cfgs)


if __name__ == '__main__':
    main()
