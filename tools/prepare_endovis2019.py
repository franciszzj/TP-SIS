# ROBUST-MIS

import os
import sys
import cv2
import json
import numpy as np

class2sents = {
    'background': [
        'background',
        'the area represented by the background tissues.',
        'background tissues are the tissues that surround the surgical site.',
    ],
    'instrument': [
        'instrument',
        'the area represented by the instrument.',
        'instruments in endoscopic surgery typically exhibit elongated designs, specialized tips or jaws for specific functions, ergonomic handles for precise control, and insulated shafts to minimize energy transmission or tissue damage.',
    ],
}


def get_one_sample(root_dir, image_file, image_path, save_dir, mask,
                   class_name):
    if '.jpg' in image_file:
        suffix = '.jpg'
    elif '.png' in image_file:
        suffix = '.png'
    mask_path = os.path.join(
        save_dir,
        image_file.replace(suffix, '') + '_{}.png'.format(class_name))
    cv2.imwrite(mask_path, mask)
    cris_data = {
        'img_path': image_path.replace(root_dir, ''),
        'mask_path': mask_path.replace(root_dir, ''),
        'num_sents': len(class2sents[class_name]),
        'sents': class2sents[class_name],
    }
    return cris_data


def process(root_dir, cris_data_file):
    cris_data_list = []
    data_type = os.getenv('DATA_TYPE', 'train')
    if data_type == 'train':
        image_dir = os.path.join(root_dir, 'Training')
        cris_masks_dir = os.path.join(root_dir, 'cris_masks', 'Training')
    elif data_type == 'test':
        image_dir = os.path.join(root_dir, 'Testing')
        cris_masks_dir = os.path.join(root_dir, 'cris_masks', 'Testing')
    if not os.path.exists(cris_masks_dir):
        os.makedirs(cris_masks_dir)
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if 'img' in image_file:
            print(image_path)
            image = cv2.imread(image_path)
            mask = cv2.imread(image_path.replace(
                '_img.png', '_label.png'))
            mask = mask[:, :, 0]

            for class_id, class_name in enumerate(['background',
                                                   'instrument']):
                if class_name == 'background':
                    target_mask = (mask == 0) * 255
                elif class_name == 'instrument':
                    target_mask = (mask > 0) * 255
                if target_mask.sum() != 0:
                    cris_data_list.append(
                        get_one_sample(root_dir, image_file, image_path,
                                       cris_masks_dir,
                                       target_mask,
                                       class_name))

    with open(os.path.join(root_dir, cris_data_file), 'w') as f:
        json.dump(cris_data_list, f)


if __name__ == '__main__':
    root_dir = sys.argv[1]
    cris_data_file = sys.argv[2]
    process(root_dir, cris_data_file)
