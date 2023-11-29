import os
import time
from tqdm import tqdm
import cv2
import copy
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
# import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def train(train_loader, model, optimizer, scheduler, scaler, epoch, cfgs):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, cfgs.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with amp.autocast():
            results = model(image, text, target)
            pred = results['pred']
            target = results['target']
            loss = results['loss']

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfgs.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfgs.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfgs.print_freq == 0:
            progress.display(i + 1)
            # if dist.get_rank() in [-1, 0]:
            #     wandb.log(
            #         {
            #             "time/batch": batch_time.val,
            #             "time/data": data_time.val,
            #             "training/lr": lr.val,
            #             "training/loss": loss_meter.val,
            #             "training/iou": iou_meter.val,
            #             "training/prec@50": pr_meter.val,
            #         },
            #         step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, cfgs):
    iou_list = []
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        # inference
        results = model(imgs, texts)
        preds = results['pred']
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        # process one batch
        for pred, mask_path, mat, ori_size in zip(preds, param['mask_path'],
                                                  param['inverse'],
                                                  param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred,
                                  mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
            if mask.shape != pred.shape:
                mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            mask = mask / 255.
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = (np.sum(inter) + 1e-6) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, cfgs.epochs, 100. * iou.item())
    logger.info(head + temp)
    return iou.item(), prec


@torch.no_grad()
def inference(test_loader, model, cfgs):
    iou_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_path'][0], flags=cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.
        # dump image & mask
        # if cfgs.visualize:
        #     seg_id = param['seg_id'][0].cpu().numpy()
        #     img_name = '{}-img.jpg'.format(seg_id)
        #     mask_name = '{}-mask.png'.format(seg_id)
        #     cv2.imwrite(filename=os.path.join(cfgs.vis_dir, img_name),
        #                 img=param['ori_img'][0].cpu().numpy())
        #     cv2.imwrite(filename=os.path.join(cfgs.vis_dir, mask_name),
        #                 img=mask)
        if cfgs.visualize:
            results_for_eval = dict()
            results_for_eval['iou_list'] = []
            results_for_eval['save_dict_list'] = []
            results_for_eval['score_name_list'] = []
        # multiple sentences
        for sent_idx, sent in enumerate(param['sents']):
            if cfgs.only_pred_first_sent and (sent_idx != 0):
                continue
            if cfgs.use_moe_select_best_sent:
                assert cfgs.only_pred_first_sent
                text_list = []
                for i_sent in range(cfgs.max_sent_num):
                    text = tokenize(
                        param['sents'][i_sent % len(param['sents'])],
                        cfgs.word_len).squeeze(0)
                    text_list.append(text)
                text = torch.stack(text_list, dim=0).unsqueeze(0)
            else:
                text = tokenize(sent, cfgs.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            results = model(img, text)
            pred = results['pred']
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            if cfgs.visualize:
                save_dict = {
                    'pred': copy.deepcopy(pred),
                    'mat': mat,
                    'h': h,
                    'w': w,
                }
                results_for_eval['save_dict_list'].append(save_dict)
            pred = cv2.warpAffine(pred,
                                  mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            if mask.shape != pred.shape:
                mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = (np.sum(inter) + 1e-6) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if cfgs.visualize:
                if 'EndoVis2017' in param['mask_path'][0]:
                    image_split = param['mask_path'][0].split('/')[-3]
                    image_id = param['mask_path'][0].split('/')[-1].split(
                        '_')[0]
                elif 'EndoVis2018' in param['mask_path'][0]:
                    image_split = '_'.join(
                        param['mask_path'][0].split('/')[-1].split('_')[:2])
                    image_id = param['mask_path'][0].split('/')[-1].split(
                        '_')[2]
                elif 'EndoVis2019' in param['mask_path'][0]:
                    image_split = param['mask_path'][0].split('/')[-2]
                    image_id = '_'.join(
                        param['mask_path'][0].split('/')[-1].split('_')[0:4])
                elif 'CholecSeg8k' in param['mask_path'][0]:
                    image_split = param['mask_path'][0].split('/')[-2]
                    image_id = '_'.join(
                        param['mask_path'][0].split('/')[-1].split('_')[0:3])
                elif 'AutoLaparo' in param['mask_path'][0]:
                    image_split = param['mask_path'][0].split('/')[-2]
                    image_id = param['mask_path'][0].split(
                        '/')[-1].split('_')[0]
                else:
                    assert False, 'not support dataset: {}'.format(
                        param['mask_path'][0])
                seg_type = '_'.join(param['sents'][0][0].split(' ')[:5])
                sent = "_".join(sent[0].split(" ")[:5])

                # save results for eval
                if cfgs.test_sents_type == 'use_best_sent_label':
                    results_for_eval['iou_list'].append(iou)
                elif cfgs.test_sents_type == 'use_best_sent_pred':
                    results_for_eval['iou_list'].append(
                        results['mask_iou_pred'].item())
                score_name = 'score-{}-{}-{}.npz'.format(
                    image_split, image_id, seg_type)
                results_for_eval['score_name_list'].append(score_name)

                # save results for vis
                pred_name = 'pred-{}-{}-{}-iou={:.2f}-{}.jpg'.format(
                    image_split, image_id, seg_type, iou * 100, sent)
                if 'EndoVis2017' in cfgs.test_data_root:
                    if 'train' in cfgs.test_data_root:
                        suffix = 'jpg'
                    elif 'test' in cfgs.test_data_root:
                        suffix = 'png'
                    image = cv2.imread(
                        os.path.join(
                            cfgs.test_data_root,
                            '{}/images/{}.{}'.format(image_split, image_id,
                                                     suffix)))
                elif 'EndoVis2018' in cfgs.test_data_root:
                    suffix = 'png'
                    image = cv2.imread(
                        os.path.join(
                            cfgs.test_data_root,
                            'images/{}_{}.{}'.format(image_split, image_id,
                                                     suffix)))
                elif 'EndoVis2019' in cfgs.test_data_root:
                    suffix = 'png'
                    image = cv2.imread(
                        os.path.join(
                            cfgs.test_data_root,
                            '{}/{}_img.{}'.format(image_split, image_id,
                                                  suffix)))
                elif 'CholecSeg8k' in cfgs.test_data_root:
                    suffix = 'png'
                    image = cv2.imread(
                        os.path.join(
                            cfgs.test_data_root,
                            '{}/{}/{}.{}'.format(image_split.split('_')[0], image_split, image_id, suffix)))
                elif 'AutoLaparo' in cfgs.test_data_root:
                    suffix = 'jpg'
                    image = cv2.imread(
                        os.path.join(
                            cfgs.test_data_root,
                            'autolaparo/imgs/{}/{}.{}'.format(image_split, image_id,
                                                              suffix)))
                show = np.zeros(image.shape)
                show[:, :, 0] = 255
                pred = pred.astype(np.float64) * 0.5
                vis_image = image * (1 - pred[:, :, None]) + \
                    show * pred[:, :, None]
                cv2.imwrite(os.path.join(cfgs.vis_dir, pred_name), vis_image)
                if cfgs.use_mae_gen_target_area:
                    mae_img = results['mae_img']
                    masked_mae_img = results['mased_mae_img']
                    mae_pred = results['mae_pred']
                    mae_img_paste = results['mae_img_paste']
                    mae_path = os.path.join(
                        cfgs.mae_vis_dir,
                        'mae-{}-{}-{}-'.format(image_split, image_id,
                                               seg_type))
                    save_img(mae_img[0], mae_path + 'original.jpg')
                    save_img(masked_mae_img[0], mae_path + 'masked.jpg')
                    save_img(mae_pred[0], mae_path + 'reconstruct.jpg')
                    save_img(mae_img_paste[0],
                             mae_path + 'reconstruct_paste.jpg')

        if cfgs.visualize:
            if cfgs.test_sents_type == 'use_class_name_sent':
                score_name = results_for_eval['score_name_list'][0]
                save_dict = results_for_eval['save_dict_list'][0]
            elif cfgs.test_sents_type in [
                    'use_best_sent_label', 'use_best_sent_pred'
            ]:
                best_idx = np.argmax(results_for_eval['iou_list'])
                score_name = results_for_eval['score_name_list'][best_idx]
                save_dict = results_for_eval['save_dict_list'][best_idx]
            np.savez_compressed(os.path.join(cfgs.score_dir, score_name),
                                **save_dict)

    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100. * iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100. * v))

    return iou.item(), prec


def save_img(image, save_path):
    if image.is_cuda:
        image = image.cpu()
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0,
                       255).int().numpy()
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
