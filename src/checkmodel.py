# from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
# #
# # import _init_paths
# #
# # import os
# #
# # import torch
# # import torch.utils.data
# # from opts import opts
# # from models.model import create_model, load_model, save_model
# # heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2,'hm_hp': 17,'hp_offset': 2}
# #
# # model = create_model('mobile_1', heads, 64)
# # # print(model)
# # path = '../exp/multi_pose/resnet_all_16/model_best.pth'
# # optimizer = torch.optim.Adam(model.parameters(), 1)
# # # model, optimizer, start_epoch = load_model(path, load_model, optimizer)
# # checkpoint = torch.load(path)
# #
# # state_dict_ = checkpoint['state_dict']
# # state_dict = {}
# #
# # # convert data_parallal to model
# # for k in state_dict_:
# #     if k.startswith('module') and not k.startswith('module_list'):
# #         state_dict[k[7:]] = state_dict_[k]
# #     else:
# #         state_dict[k] = state_dict_[k]
# # model_state_dict = model.state_dict()
# #
# # # check loaded parameters and created model parameters
# # for k in state_dict:
# #     if k in model_state_dict:
# #         if state_dict[k].shape != model_state_dict[k].shape:
# #             print('Skip loading parameter {}, required shape{}, ' \
# #                   'loaded shape{}.'.format(
# #                 k, model_state_dict[k].shape, state_dict[k].shape))
# #             state_dict[k] = model_state_dict[k]
# #     else:
# #         print('Drop parameter {}.'.format(k))
# # for k in model_state_dict:
# #     if not (k in state_dict):
# #         print('No param {}.'.format(k))
# #         state_dict[k] = model_state_dict[k]
# # model.load_state_dict(state_dict, strict=False)
# #
# # print(1)

import torch
from lib.models.networks.mobilenet import *
from torchvision.models.mobilenet import mobilenet_v2

heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2,'hm_hp': 17,'hp_offset': 2}

path = '/home/wuxilab/dxx/CenterNet/src/mobilenet_v2.pth.tar'
net = MobileNetV2(InvertedResidual, heads, 64, path=path)



debug=1

