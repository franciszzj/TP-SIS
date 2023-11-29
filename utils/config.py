# -----------------------------------------------------------------------------
# Functions for parsing args
# -----------------------------------------------------------------------------
import copy
import os
from ast import literal_eval

import yaml


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    # pre-difined cfgs
    cfg = {
        # DATA
        'dataset': 'endovis2017',
        'train_data_file': 'cris_train.json',
        'train_data_root': './EndoVis2017/cropped_train/',
        'val_data_file': 'cris_test.json',
        'val_data_root': './EndoVis2017/cropped_test/',
        'sents_select_type': 'random',
        'use_vis_aug': False,
        'use_vis_aug_non_rigid': False,
        # TRAIN
        'freeze_modules': [],
        ## Base Arch
        'clip_pretrain': 'pretrain/RN50.pt',
        'input_size': 416,
        'word_len': 17,
        'word_dim': 1024,
        'vis_dim': 512,
        'fpn_in': [512, 1024, 1024],
        'fpn_out': [256, 512, 1024],
        'sync_bn': True,
        ## Neck
        'neck_with_text_state': True,
        ## Decoder
        'num_layers': 3,
        'num_head': 8,
        'dim_ffn': 2048,
        'dropout': 0.1,
        'intermediate': False,
        ## MaskIoU
        'pred_mask_iou': False,
        'mask_iou_loss_type': 'mse',
        'mask_iou_loss_weight': 1.0,
        ## MoE
        'use_moe_select_best_sent': False,
        'max_sent_num': 7,
        'moe_selector_type': 'best',  # best, weighted_sum
        'use_moe_consistency_loss': False,
        'moe_consistency_loss_weight': 1.0,
        ## MAE
        'use_mae_gen_target_area': False,
        'mae_pretrain': 'pretrain/mae_pretrain_vit_base.pth',
        'mae_input_shape': (224, 224),
        'mae_mask_ratio': 0.75,
        'reconstruct_full_img': False,
        'mae_hard_example_mining_type': None,
        'mae_shared_encoder': False,
        ## Training Setting
        'workers': 8,  # data loader workers
        'workers_val': 4,
        'epochs': 50,
        'milestones': [35],
        'start_epoch': 0,
        'batch_size': 64,  # batch size for training
        # batch size for validation during training, memory and speed tradeoff
        'batch_size_val': 64,
        'optimizer': 'adam',
        'base_lr': 0.0001,
        'lr_decay': 0.1,
        'lr_multi': 0.1,
        'weight_decay': 0.,
        'amsgrad': False,
        'max_norm': 0.,
        'manual_seed': 0,
        'print_freq': 100,
        ## Resume & Save
        'exp_name': 'CRIS_R50',
        'output_folder': 'exp/endovis2017',
        'save_freq': 1,
        'weight': None,  # path to initial weight (default: none)
        'resume': None,  # path to latest checkpoint (default: none)
        # evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend
        'evaluate': True,
        # Distributed
        'dist_url': 'tcp://localhost:3681',
        'dist_backend': 'nccl',
        'multiprocessing_distributed': True,
        'world_size': 1,
        'rank': 0,
        # TEST
        'test_data_file': 'cris_test.json',
        'test_data_root': './EndoVis2017/cropped_test/',
        'visualize': False,
    }
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            assert k in cfg.keys(), \
                'config must be pre-defined, but get: {}'.format(k)
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(value, cfg[subkey], subkey,
                                                 full_key)
        setattr(new_cfg, subkey, value)

    return new_cfg


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(original_type, replacement_type, original,
                         replacement, full_key))
