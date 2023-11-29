import os
import sys
import cv2
import json
import numpy as np

class_list = ['black_background', 'abdominal_wall', 'liver',
              'gastrointestinal_tract', 'fat', 'grasper',
              'connective_tissue', 'blood', 'cystic_duct',
              'l_hook_electrocautery', 'gallbladder', 'hepatic_vein',
              'liver_ligament']

# All
video_dir_list = ['video01', 'video09', 'video12', 'video17', 'video18', 'video20',
                  'video24', 'video25', 'video26', 'video27', 'video28', 'video35',
                  'video37', 'video43', 'video48', 'video52', 'video55']

# A spatio-temporal network for video semantic segmentation in surgical videos
# train_video_dir_list = ['video01', 'video09', 'video17', 'video18', 'video24',
#                         'video25', 'video26', 'video27', 'video28', 'video35',
#                         'video37', 'video43', 'video52']
# val_video_dir_list = []
# test_video_dir_list = ['video12', 'video20', 'video48', 'video55']

# Class-wise confidence-aware active learning for laparoscopic images segmentation
train_video_dir_list = ['video01', 'video09', 'video17', 'video18', 'video20',
                        'video24', 'video25', 'video26', 'video27', 'video28', 'video35',
                        'video37', 'video43']
val_video_dir_list = []
test_video_dir_list = ['video12', 'video48', 'video52', 'video55']

# Analysis of Current Deep Learning Networks for Semantic Segmentation of Anatomical Structures in Laparoscopic Surgery
# train_video_dir_list = ['video01', 'video09', 'video18', 'video20', 'video24',
#                         'video25', 'video26', 'video28', 'video35', 'video37',
#                         'video43', 'video48', 'video55', ]
# val_video_dir_list = ['video17', 'video52']
# test_video_dir_list = ['video12', 'video27']

class2sents = {
    'black_background': [
        'black background',
        'the area represented by the black background.',
        'the black background in the endoscopic view serves as a contrasting backdrop, highlighting the illuminated tissues and structures during surgery.',
    ],
    'abdominal_wall': [
        'abdominal wall',
        'the area represented by the abdominal wall.',
        'the abdominal wall displays layered tissues, including skin, fat, fascia, and muscles, with visible blood vessels beneath the skin and a glistening white from the fibrous connective tissue, presenting a reddish-pink hue.',
    ],
    'liver': [
        'liver',
        'the area represented by the liver.',
        'the liver emerges as a large, reddish-brown organ with a smooth, glistening texture, showcasing its vasculature and, at times, its lobulated edges, and it feels firm but slightly pliable when manipulated.',
    ],
    'gastrointestinal_tract': [
        'gastrointestinal tract',
        'the area represented by the gastrointestinal tract.',
        'the gastrointestinal tract appears as a long tube, varying in diameter and lining, displaying a pink to reddish inner surface with distinctive features such as folds, villi, or movements like peristalsis, and specific structures like the stomach\'s rugae or the colon\'s haustra can be discerned.',
    ],
    'fat': [
        'fat',
        'the area represented by the fat.',
        'fat is recognizable by its yellowish, soft, and lobulated texture, setting itself apart from adjacent tissues due to its unique color and malleability.',
    ],
    'grasper': [
        'grasper',
        'the area represented by the grasper.',
        'the grasper shines with its metallic surface, characterized by a long, slender shaft and pincer-like ends, making it distinguishable and designed to efficiently grasp and hold tissues.',
    ],
    'connective_tissue': [
        'connective tissue',
        'the area represented by the connective tissue.',
        'connective tissue manifests as fibrous, occasionally glistening structures, varying in shades from white to translucent or slightly yellowish, serving the pivotal role of anchoring organs or surrounding vessels and nerves.',
    ],
    'blood': [
        'blood',
        'the area represented by the blood.',
        'blood appears as a fluid ranging in color from bright red to dark maroon, with pools or droplets occasionally forming on tissues, and sites of active bleeding may exhibit spurting or oozing patterns.',
    ],
    'cystic_duct': [
        'cystic duct',
        'the area represented by the cystic duct.',
        'the cystic duct emerges as a whitish or pale yellow tubular conduit linking the gallbladder to the common bile duct, with its distinct hue and surrounding connective tissue setting it apart from nearby structures.',
    ],
    'l_hook_electrocautery': [
        'l-hook electrocautery',
        'the area represented by the l-hook electrocautery.',
        'the l-hook electrocautery showcases a slender metallic form with a curved tip, and when energized, it emits a localized electrical arc or spark for tissue cutting or coagulation.',
    ],
    'gallbladder': [
        'gallbladder',
        'the area represented by the gallbladder.',
        'the gallbladder presents itself as a pear-shaped, greenish-brown sac under the liver, with a texture ranging from smooth to slightly irregular, and it may contain bile of differing viscosities and shades.',
    ],
    'hepatic_vein': [
        'hepatic vein',
        'the area represented by the hepatic vein.',
        'the hepatic vein emerges as a prominent vascular conduit arising from the liver and heading toward the inferior vena cava, with smooth walls that are typically darker than the adjacent liver tissue.',
    ],
    'liver_ligament': [
        'liver ligament',
        'the area represented by the liver ligament.',
        'the liver ligament stands out as a thin, fibrous band with a firmer texture than the nearby liver tissue, reflecting light distinctively due to its connective nature.',

    ],
}


class2rgb = {
    'black_background': (50, 50, 50),
    'abdominal_wall': (11, 11, 11),
    'liver': (21, 21, 21),
    'gastrointestinal_tract': (13, 13, 13),
    'fat': (12, 12, 12),
    'grasper': (31, 31, 31),
    'connective_tissue': (23, 23, 23),
    'blood': (24, 24, 24),
    'cystic_duct': (25, 25, 25),
    'l_hook_electrocautery': (32, 32, 32),
    'gallbladder': (22, 22, 22),
    'hepatic_vein': (33, 33, 33),
    'liver_ligament': (5, 5, 5),
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
        video_dir_list = train_video_dir_list
    elif data_type == 'val':
        video_dir_list = val_video_dir_list
    elif data_type == 'test':
        video_dir_list = test_video_dir_list
    for video_dir in video_dir_list:
        for image_dir in os.listdir(os.path.join(root_dir, video_dir)):
            print(os.path.join(root_dir, video_dir, image_dir))
            cris_masks_dir = os.path.join(
                root_dir, 'cris_masks', video_dir, image_dir)
            if not os.path.exists(cris_masks_dir):
                os.makedirs(cris_masks_dir)
            for image_file in os.listdir(os.path.join(root_dir, video_dir, image_dir)):
                image_path = os.path.join(
                    root_dir, video_dir, image_dir, image_file)
                if 'mask' not in image_file:
                    image = cv2.imread(image_path)
                    mask = cv2.imread(image_path.replace(
                        '.png', '_watershed_mask.png'))
                    mask = mask[:, :, 0]

                    for class_id, class_name in enumerate(class_list):
                        target_mask = (mask == class2rgb[class_name][0]) * 255
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
