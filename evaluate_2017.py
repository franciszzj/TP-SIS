import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instruments_factor = 32


def ch_iou(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in set(y_true.flatten()):
        if type_id == 0:
            continue
        result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)


def isi_iou(y_true, y_pred, problem_type='binary'):
    result = []

    if problem_type == 'binary':
        type_number = 2
    elif problem_type == 'parts':
        type_number = 4
    elif problem_type == 'instruments':
        type_number = 8

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in range(type_number):
        if type_id == 0:
            continue
        if (y_true == type_id).sum() != 0 or (y_pred == type_id).sum() != 0:
            result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)


def mc_iou(y_true, y_pred, problem_type='binary'):
    result = []

    if problem_type == 'binary':
        type_number = 2
    elif problem_type == 'parts':
        type_number = 4
    elif problem_type == 'instruments':
        type_number = 8

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return [-1 for _ in range(type_number - 1)]
        else:
            return [0 for _ in range(type_number - 1)]

    for type_id in range(type_number):
        if type_id == 0:
            continue
        if (y_true == type_id).sum() != 0 or (y_pred == type_id).sum() != 0:
            result += [iou(y_true == type_id, y_pred == type_id)]
        else:
            result += [-1]
    return result


def ch_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in set(y_true.flatten()):
        if type_id == 0:
            continue
        result += [dice(y_true == type_id, y_pred == type_id)]

    return np.mean(result)


def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-6) / (y_true.sum() +
                                                   y_pred.sum() + 1e-6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--test_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test',
        help='path where train images with ground truth are located')
    arg('--pred_path',
        type=str,
        default=
        '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/CRIS/exp/endovis2017/v1_0/score',
        help='path with predictions')
    arg('--problem_type',
        type=str,
        default='parts',
        choices=['binary', 'parts', 'instruments'])
    arg('--vis', action='store_true')
    args = parser.parse_args()

    if args.problem_type == 'binary':
        class_name_list = ['background', 'instrument']
        factor = binary_factor
    elif args.problem_type == 'parts':
        class_name_list = ['background', 'shaft', 'wrist', 'claspers']
        factor = parts_factor
    elif args.problem_type == 'instruments':
        class_name_list = [
            'background', 'bipolar_forceps', 'prograsp_forceps',
            'large_needle_driver', 'vessel_sealer', 'grasping_retractor',
            'monopolar_curved_scissors', 'other_medical_instruments'
        ]
        factor = instruments_factor
    class_name_compatible = {
        'other_medical_instruments': 'ultrasound_probe',
        'ultrasound_probe': 'other_medical_instruments',
    }

    result_ch_iou = []
    result_ch_dice = []
    result_isi_iou = []
    result_mc_iou = []

    # palette
    if args.vis:
        eval_dir = os.path.join(args.pred_path.replace('/score', '/eval_vis'))
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        palette_list = [(255, 128, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (0, 255, 255), (255, 0, 255), (255, 255, 0),
                        (0, 128, 255)]
        palette = np.zeros((8, height, width, 3))
        for i in range(8):
            for j in range(3):
                palette[i][:, :, j] = palette_list[i][j]

    # evaluate
    if 'train' in args.test_path:
        dataset_num = 8
    elif 'test' in args.test_path:
        dataset_num = 10
    for instrument_id in range(1, dataset_num + 1):
        instrument_dataset_name = 'instrument_dataset_{}'.format(instrument_id)
        file_dir = os.path.join(args.test_path, instrument_dataset_name,
                                '{}_masks'.format(args.problem_type))
        if args.vis:
            image_dir = os.path.join(args.test_path, instrument_dataset_name,
                                     'images')
        for file_name in tqdm(os.listdir(file_dir),
                              desc=instrument_dataset_name):
            file_id = file_name.split('.')[0]

            file_path = os.path.join(file_dir, file_name)
            y_true = cv2.imread(file_path, 0).astype(np.uint8)
            y_true = y_true // factor

            if args.vis:
                if 'cropped_train' in args.test_path:
                    image_path = os.path.join(image_dir,
                                              '{}.jpg'.format(file_id))
                elif 'cropped_test' in args.test_path:
                    image_path = os.path.join(image_dir,
                                              '{}.png'.format(file_id))
                image = cv2.imread(image_path)

                show = np.zeros_like(image)
                for i_h in range(height):
                    for i_w in range(width):
                        show[i_h, i_w] = palette[y_true[i_h, i_w], i_h, i_w]
                # show = np.take(palette, gt_mask)
                gt_vis_image = image * 0.5 + show * 0.5

            pred_image_list = []
            for class_name in class_name_list:
                pred_file_name = os.path.join(
                    args.pred_path,
                    'score-{}-{}-{}.npz'.format(instrument_dataset_name,
                                                file_id, class_name))
                if class_name in class_name_compatible.keys():
                    if not os.path.exists(pred_file_name):
                        pred_file_name = pred_file_name.replace(
                            class_name, class_name_compatible[class_name])
                if os.path.exists(pred_file_name):
                    pred_dict = np.load(pred_file_name)
                    pred_image = cv2.warpAffine(pred_dict.get('pred'),
                                                pred_dict.get('mat'),
                                                (width, height),
                                                flags=cv2.INTER_CUBIC,
                                                borderValue=0.)
                else:
                    pred_image = np.zeros_like(y_true)
                pred_image_list.append(pred_image)
            pred_image = np.array(pred_image_list)
            y_pred = np.argmax(pred_image, axis=0)

            if args.vis:
                show = np.zeros_like(image)
                for i_h in range(height):
                    for i_w in range(width):
                        show[i_h, i_w] = palette[y_pred[i_h, i_w], i_h, i_w]
                pred_vis_image = image * 0.5 + show * 0.5

                vis_image = np.concatenate([gt_vis_image, pred_vis_image],
                                           axis=1)
                cv2.imwrite(
                    '{}/{}_{}_{}.jpg'.format(eval_dir, args.problem_type,
                                             instrument_dataset_name, file_id),
                    vis_image)

            y_pred = y_pred.astype(np.uint8)

            # Challenge IoU
            result_ch_iou += [ch_iou(y_true, y_pred)]
            result_ch_dice += [ch_dice(y_true, y_pred)]
            # ISI IoU
            result_isi_iou += [isi_iou(y_true, y_pred, args.problem_type)]
            # Mean Class IoU
            result_mc_iou += [mc_iou(y_true, y_pred, args.problem_type)]

    print('Ch IoU: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_ch_iou) * 100, np.std(result_ch_iou)))
    print('Ch Dice: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_ch_dice) * 100, np.std(result_ch_dice)))
    print('ISI IoU: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_isi_iou) * 100, np.std(result_isi_iou)))
    result_mc = []
    for c, class_name in enumerate(class_name_list[1:]):
        result_c = []
        for n in range(len(result_mc_iou)):
            if result_mc_iou[n][c] >= 0:
                result_c.append(result_mc_iou[n][c])
        print('Instrument Class: {} IoU={:.2f}, std={:.4f}'.format(
            class_name,
            np.mean(result_c) * 100, np.std(result_c)))
        result_mc.append(np.mean(result_c))
    print('MC IoU: mean={:.2f}, std={:.4f}'.format(
        np.mean(result_mc) * 100, np.std(result_mc)))
