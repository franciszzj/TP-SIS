from .segmenter import CRIS
from loguru import logger

# def build_segmenter(cfgs):
#     model = CRIS(cfgs)
#     backbone = []
#     backbone_no_decay = []
#     head = []
#     for k, v in model.named_parameters():
#         if k.startswith('backbone') and 'positional_embedding' not in k:
#             backbone.append(v)
#         elif 'positional_embedding' in k:
#             backbone_no_decay.append(v)
#         else:
#             head.append(v)
#     print('Backbone with decay: {}, Backbone without decay: {}, Head: {}'.format(
#         len(backbone), len(backbone_no_decay), len(head)))
#     param_list = [{
#         'params': backbone,
#         'initial_lr': cfgs.lr_multi * cfgs.base_lr
#     }, {
#         'params': backbone_no_decay,
#         'initial_lr': cfgs.lr_multi * cfgs.base_lr,
#         'weight_decay': 0
#     }, {
#         'params': head,
#         'initial_lr': cfgs.base_lr
#     }]
#     return model, param_list


def build_segmenter(cfgs):
    model = CRIS(cfgs)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': cfgs.lr_multi * cfgs.base_lr
    }, {
        'params': head,
        'initial_lr': cfgs.base_lr
    }]
    return model, param_list
