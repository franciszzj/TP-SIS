import os
import sys
import cv2
import json
import numpy as np

class2sents = {
    'background': [
        'background',
        'the area represented by the black background.',
        'background tissues are the tissues that surround the surgical site.',
    ],
    'instrument': [
        'instrument',
        'the area represented by the abdominal wall.',
        'instruments in endoscopic surgery typically exhibit elongated designs, specialized tips or jaws for specific functions, ergonomic handles for precise control, and insulated shafts to minimize energy transmission or tissue damage.',
    ],
    'shaft': [
        'shaft',
        'the area represented by the shaft of the instrument.',
        'the shaft of an instrument in endoscopic surgery generally appears as a long, slender structure, often insulated, providing a connection between the ergonomic handle and the specialized tip or jaw for precise surgical maneuvers.',
    ],
    'manipulator': [
        'manipulator',
        'the area represented by the manipulator of the instrument.',
        'the manipulator shines with its metallic surface, characterized by a sleek, elongated shaft and specialized ends, designed to move or adjust internal tissues or organs, contrasting starkly with the body\'s organic textures.',
    ],
    'grasping_forceps': [
        'grasping forceps',
        'the area represented by the grasping forceps.',
        'grasping forceps emerge as long, slender, metallic instruments with clamping jaws at the tip, intended for grasping and retrieval, and their reflective body contrasts notably with the surrounding soft tissues.',
    ],
    'ligasure': [
        'ligasure',
        'the area represented by the ligasure.',
        'the Ligasure stands out with its sleek, elongated design and specialized tips, showcasing a blend of metal and plastic materials, designed specifically for tissue coagulation and division, offering a distinct contrast to the body\'s organic structures.',
    ],
    'dissecting_and_grasping_forceps': [
        'dissecting and grasping forceps',
        'the area represented by the dissecting and grasping forceps.',
        'dissecting and grasping forceps present with elongated metallic forms, where the former showcases precise, slender jaws for delicate tissue separation, and the latter displays broader jaws for holding or pinching, both standing out distinctly amidst the organic backdrop.',
    ],
    'electric_hook': [
        'electric hook',
        'the area represented by the electric hook.',
        'the electric hook emerges with a slender metallic form and a curved, pointed tip, capable of producing an electrical arc when activated, offering a noticeable contrast to the body\'s softer internal textures.',
    ],
    'uterus': [
        'uterus',
        'the area represented by the uterus.',
        'the uterus presents itself as a muscular, pear-shaped structure with a surface ranging from smooth to slightly textured, and its reddish-pink coloration can fluctuate based on blood circulation.',
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
        img_dir = os.path.join(root_dir, 'autolaparo/imgs/train')
        cris_masks_dir = os.path.join(
            root_dir, 'cris_masks/autolaparo/imgs/train')
    elif data_type == 'val':
        img_dir = os.path.join(root_dir, 'autolaparo/imgs/val')
        cris_masks_dir = os.path.join(
            root_dir, 'cris_masks/autolaparo/imgs/val')
    elif data_type == 'test':
        img_dir = os.path.join(root_dir, 'autolaparo/imgs/test')
        cris_masks_dir = os.path.join(
            root_dir, 'cris_masks/autolaparo/imgs/test')
    if not os.path.exists(cris_masks_dir):
        os.makedirs(cris_masks_dir)
    for image_file in os.listdir(img_dir):
        image_path = os.path.join(img_dir, image_file)
        print(image_path)
        image = cv2.imread(image_path)
        mask_path = image_path.replace(
            'autolaparo/imgs', 'autolaparo/masks').replace('.jpg', '.png')
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0]

        for class_id, class_name in enumerate(class2sents.keys()):
            if class_name == 'background':
                target_mask = (mask == 0) * 255
            elif class_name == 'instrument':
                target_mask = np.logical_and((mask != 0), (mask != 180)) * 255
            elif class_name == 'shaft':
                target_mask = np.logical_or(np.logical_or(
                    (mask == 40), (mask == 80)), np.logical_or((mask == 120), (mask == 160))) * 255
            elif class_name == 'manipulator':
                target_mask = np.logical_or(np.logical_or(
                    (mask == 20), (mask == 60)), np.logical_or((mask == 100), (mask == 140))) * 255
            elif class_name == 'grasping_forceps':
                target_mask = np.logical_or((mask == 20), (mask == 40)) * 255
            elif class_name == 'ligasure':
                target_mask = np.logical_or((mask == 60), (mask == 80)) * 255
            elif class_name == 'dissecting_and_grasping_forceps':
                target_mask = np.logical_or((mask == 100), (mask == 120)) * 255
            elif class_name == 'electric_hook':
                target_mask = np.logical_or((mask == 140), (mask == 160)) * 255
            elif class_name == 'uterus':
                target_mask = (mask == 180) * 255
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
