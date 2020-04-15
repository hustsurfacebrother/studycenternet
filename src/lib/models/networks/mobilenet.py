#
# ## pretrained_256deconv
#
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Dequan Wang and Xingyi Zhou
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import math
# import logging
#
# import torch
# import torch.nn as nn
# # from .DCNv2.dcn_v2 import DCN
# import torch.utils.model_zoo as model_zoo
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, std=0.001)
#             # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             # torch.nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
#
#
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
#         self.inplanes = 1280
#         self.deconv_with_bias = False
#         self.heads = heads
#         super(MobileNetV2, self).__init__()
#
#         block = block
#         input_channel = 32
#         last_channel = 1280
#
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # used for deconv layers
#         llast = 256
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [256, 256, llast ],
#             [4, 4, 4],
#         )
#         # self.final_layer = []
#         wh_hp = 64
#         for head in sorted(self.heads):
#             num_output = self.heads[head]
#             if head_conv > 0:
#
#
#                 # if head == 'wh':
#                 #     fc = nn.Sequential(
#                 #         nn.Conv2d(self.heads['hm'], wh_hp,
#                 #                   kernel_size=3, padding=1, bias=True),
#                 #         nn.ReLU(inplace=True),
#                 #         nn.Conv2d(wh_hp, num_output,
#                 #                   kernel_size=1, stride=1, padding=0))
#                 # elif head == 'hps':
#                 #     fc = nn.Sequential(
#                 #         nn.Conv2d(self.heads['hm_hp'], wh_hp,
#                 #                   kernel_size=3, padding=1, bias=True),
#                 #         nn.ReLU(inplace=True),
#                 #         nn.Conv2d(wh_hp, num_output,
#                 #                   kernel_size=1, stride=1, padding=0))
#                 # else:
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast, head_conv,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(head_conv, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#             else:
#
#                 fc = nn.Conv2d(
#                     in_channels=llast,
#                     out_channels=num_output,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0
#                 )
#             self.__setattr__(head, fc)
#
#
#
#
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#
#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)
#
#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         # x = x.mean(3).mean(2)
#
#         x = self.deconv_layers(x)
#
#
#         ret = {}
#
#         for head in self.heads:
#             ret[head] = self.__getattr__(head)(x)
#
#
#
#
#         return [ret]
#
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             padding = 1
#             output_padding = 0
#
#
#         return deconv_kernel, padding, output_padding
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()
#
#     def _initialize_weights(self,path):
#         if path!=None:
#             # print('=> init resnet deconv weights from normal distribution')
#             for _, m in self.deconv_layers.named_modules():
#                 if isinstance(m, nn.ConvTranspose2d):
#                     # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.normal_(m.weight, std=0.001)
#                     if self.deconv_with_bias:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     # print('=> init {}.weight as 1'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             # print('=> init final conv weights from normal distribution')
#             for head in self.heads:
#                 final_layer = self.__getattr__(head)
#                 for i, m in enumerate(final_layer.modules()):
#                     if isinstance(m, nn.Conv2d):
#                         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                         # print('=> init {}.bias as 0'.format(name))
#                         if m.weight.shape[0] == self.heads[head]:
#                             if 'hm' in head:
#                                 nn.init.constant_(m.bias, -2.19)
#                             else:
#                                 nn.init.normal_(m.weight, std=0.001)
#                                 nn.init.constant_(m.bias, 0)
#             pretrained_dict = torch.load(path)
#             mobilenet_state_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict}
#             mobilenet_state_dict.update(pretrained_dict)
#             self.load_state_dict(mobilenet_state_dict, strict=False)
#
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#                 elif isinstance(m, nn.Linear):
#                     n = m.weight.size(1)
#                     m.weight.data.normal_(0, 0.01)
#                     m.bias.data.zero_()
#
# def get_pose_net(heads, head_conv,num_layers=1):
#     path ='/home/wuxilab/dxx/CenterNet/src/mobilenet_v2.pth.tar'
#     model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
#     model._initialize_weights(path=path)
#
#     return model
#



# ## pretrained_256deconv_with focal to wh
#
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Dequan Wang and Xingyi Zhou
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import math
# import logging
#
# import torch
# import torch.nn as nn
# # from .DCNv2.dcn_v2 import DCN
# import torch.utils.model_zoo as model_zoo
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, std=0.001)
#             # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             # torch.nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
#         self.inplanes = 1280
#         self.deconv_with_bias = False
#         self.heads = heads
#         super(MobileNetV2, self).__init__()
#
#         block = block
#         input_channel = 32
#         last_channel = 1280
#
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # used for deconv layers
#         llast = 256
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [256, 256, llast ],
#             [4, 4, 4],
#         )
#         # self.final_layer = []
#         wh_hp = 64
#         for head in sorted(self.heads):
#             num_output = self.heads[head]
#             if head_conv > 0:
#
#
#                 if head == 'wh':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 elif head == 'hps':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 else:
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast, head_conv,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(head_conv, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#             else:
#
#                 fc = nn.Conv2d(
#                     in_channels=llast,
#                     out_channels=num_output,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0
#                 )
#             self.__setattr__(head, fc)
#
#
#
#
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#
#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)
#
#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         # x = x.mean(3).mean(2)
#
#         x = self.deconv_layers(x)
#
#         hm_x = self.hm(x)
#         hm_hp_x = self.hm_hp(x)
#         hm_hp_all = torch.cat([hm_hp_x, hm_x],1)
#         wh = self.wh(hm_hp_all)
#         hps = self.hps(hm_hp_all)
#         hp_offset = self.hp_offset(x)
#         reg = self.reg(x)
#
#
#         ret = {'hm':hm_x, 'hm_hp':hm_hp_x, 'wh':wh, 'hps':hps, 'reg':reg,
#                'hp_offset':hp_offset}
#
#
#
#
#         return [ret]
#
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             padding = 1
#             output_padding = 0
#
#
#         return deconv_kernel, padding, output_padding
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()
#
#     def _initialize_weights(self,path):
#         if path!=None:
#             # print('=> init resnet deconv weights from normal distribution')
#             for _, m in self.deconv_layers.named_modules():
#                 if isinstance(m, nn.ConvTranspose2d):
#                     # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.normal_(m.weight, std=0.001)
#                     if self.deconv_with_bias:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     # print('=> init {}.weight as 1'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             # print('=> init final conv weights from normal distribution')
#             for head in self.heads:
#                 final_layer = self.__getattr__(head)
#                 for i, m in enumerate(final_layer.modules()):
#                     if isinstance(m, nn.Conv2d):
#                         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                         # print('=> init {}.bias as 0'.format(name))
#                         if m.weight.shape[0] == self.heads[head]:
#                             if 'hm' in head:
#                                 nn.init.constant_(m.bias, -2.19)
#                             else:
#                                 nn.init.normal_(m.weight, std=0.001)
#                                 nn.init.constant_(m.bias, 0)
#             pretrained_dict = torch.load(path)
#             mobilenet_state_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict}
#             mobilenet_state_dict.update(pretrained_dict)
#             self.load_state_dict(mobilenet_state_dict, strict=False)
#
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#                 elif isinstance(m, nn.Linear):
#                     n = m.weight.size(1)
#                     m.weight.data.normal_(0, 0.01)
#                     m.bias.data.zero_()
#
# def get_pose_net(heads, head_conv,num_layers=1):
#     path ='/home/wuxilab/dxx/CenterNet/src/mobilenet_v2.pth.tar'
#     model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
#     model._initialize_weights(path=path)
#
#     return model
#


# ## pretrained_256deconv_only for focal
#
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Dequan Wang and Xingyi Zhou
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import math
# import logging
#
# import torch
# import torch.nn as nn
# # from .DCNv2.dcn_v2 import DCN
# import torch.utils.model_zoo as model_zoo
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, std=0.001)
#             # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             # torch.nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
#         self.inplanes = 1280
#         self.deconv_with_bias = False
#         self.heads = heads
#         super(MobileNetV2, self).__init__()
#
#         block = block
#         input_channel = 32
#         last_channel = 1280
#
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # used for deconv layers
#         llast = 256
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [256, 256, llast ],
#             [4, 4, 4],
#         )
#         # self.final_layer = []
#         wh_hp = 64
#         for head in sorted(self.heads):
#             num_output = self.heads[head]
#             if head_conv > 0:
#
#
#                 if head == 'wh':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 elif head == 'hps':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 else:
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast, head_conv,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(head_conv, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#             else:
#
#                 fc = nn.Conv2d(
#                     in_channels=llast,
#                     out_channels=num_output,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0
#                 )
#             self.__setattr__(head, fc)
#
#
#
#
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#
#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)
#
#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         # x = x.mean(3).mean(2)
#
#         x = self.deconv_layers(x)
#
#         hm_x = self.hm(x)
#         hm_hp_x = self.hm_hp(x)
#         # hm_hp_all = torch.cat([hm_hp_x, hm_x],1)
#         # wh = self.wh(hm_hp_all)
#         # hps = self.hps(hm_hp_all)
#         # hp_offset = self.hp_offset(x)
#         # reg = self.reg(x)
#
#
#         ret = {'hm':hm_x, 'hm_hp':hm_hp_x}
#
#
#
#
#         return [ret]
#
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             padding = 1
#             output_padding = 0
#
#
#         return deconv_kernel, padding, output_padding
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()
#
#     def _initialize_weights(self,path):
#         if path!=None:
#             # print('=> init resnet deconv weights from normal distribution')
#             for _, m in self.deconv_layers.named_modules():
#                 if isinstance(m, nn.ConvTranspose2d):
#                     # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.normal_(m.weight, std=0.001)
#                     if self.deconv_with_bias:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     # print('=> init {}.weight as 1'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             # print('=> init final conv weights from normal distribution')
#             for head in self.heads:
#                 final_layer = self.__getattr__(head)
#                 for i, m in enumerate(final_layer.modules()):
#                     if isinstance(m, nn.Conv2d):
#                         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                         # print('=> init {}.bias as 0'.format(name))
#                         if m.weight.shape[0] == self.heads[head]:
#                             if 'hm' in head:
#                                 nn.init.constant_(m.bias, -2.19)
#                             else:
#                                 nn.init.normal_(m.weight, std=0.001)
#                                 nn.init.constant_(m.bias, 0)
#             pretrained_dict = torch.load(path)
#             mobilenet_state_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict}
#             mobilenet_state_dict.update(pretrained_dict)
#             self.load_state_dict(mobilenet_state_dict, strict=False)
#
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#                 elif isinstance(m, nn.Linear):
#                     n = m.weight.size(1)
#                     m.weight.data.normal_(0, 0.01)
#                     m.bias.data.zero_()
#
# def get_pose_net(heads, head_conv,num_layers=1):
#     path ='/home/wuxilab/dxx/CenterNet/src/mobilenet_v2.pth.tar'
#     model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
#     model._initialize_weights(path=path)
#
#     return model
#

#
# ## pretrained_256deconv_先focalh后全部
#
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Dequan Wang and Xingyi Zhou
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import math
# import logging
#
# import torch
# import torch.nn as nn
# # from .DCNv2.dcn_v2 import DCN
# import torch.utils.model_zoo as model_zoo
#
# def _sigmoid(x):
#   y = torch.clamp(torch.sigmoid(x)  , min=1e-4, max=1-1e-4)
#   return y
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, std=0.001)
#             # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             # torch.nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
#         self.inplanes = 1280
#         self.deconv_with_bias = False
#         self.heads = heads
#         super(MobileNetV2, self).__init__()
#
#         block = block
#         input_channel = 32
#         last_channel = 1280
#
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # used for deconv layers
#         llast = 256
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [256, 256, llast ],
#             [4, 4, 4],
#         )
#         # self.final_layer = []
#         wh_hp = 64
#         for head in sorted(self.heads):
#             num_output = self.heads[head]
#             if head_conv > 0:
#
#
#                 if head == 'wh':
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast+18, wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 elif head == 'hps':
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast+18, wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 else:
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast, head_conv,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(head_conv, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#             else:
#
#                 fc = nn.Conv2d(
#                     in_channels=llast,
#                     out_channels=num_output,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0
#                 )
#             self.__setattr__(head, fc)
#
#
#
#
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#
#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)
#
#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#
#         x = self.deconv_layers(x)
#
#         hm_x = self.hm(x)
#         hm_hp_x = self.hm_hp(x)
#         hm_hp_all = torch.cat([hm_hp_x , hm_x, x],1)
#         wh = self.wh(hm_hp_all)
#         hps = self.hps(hm_hp_all)
#         hp_offset = self.hp_offset(x)
#         reg = self.reg(x)
#
#         ret = {'hm': hm_x, 'hm_hp': hm_hp_x, 'wh': wh, 'hps': hps, 'reg': reg,
#                        'hp_offset':hp_offset}
#
#
#
#
#         return [ret]
#
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             padding = 1
#             output_padding = 0
#
#
#         return deconv_kernel, padding, output_padding
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()
#
#     def _initialize_weights(self,path):
#         if path!=None:
#             # print('=> init resnet deconv weights from normal distribution')
#             for _, m in self.deconv_layers.named_modules():
#                 if isinstance(m, nn.ConvTranspose2d):
#                     # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.normal_(m.weight, std=0.001)
#                     if self.deconv_with_bias:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     # print('=> init {}.weight as 1'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             # print('=> init final conv weights from normal distribution')
#             for head in self.heads:
#                 final_layer = self.__getattr__(head)
#                 for i, m in enumerate(final_layer.modules()):
#                     if isinstance(m, nn.Conv2d):
#                         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                         # print('=> init {}.bias as 0'.format(name))
#                         if m.weight.shape[0] == self.heads[head]:
#                             if 'hm' in head:
#                                 nn.init.constant_(m.bias, -2.19)
#                             else:
#                                 nn.init.normal_(m.weight, std=0.001)
#                                 nn.init.constant_(m.bias, 0)
#             pretrained_model = torch.load(path, map_location= lambda storage, loc: storage)
#             pretrained_dict = pretrained_model['state_dict']
#             mobilenet_state_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict and k[0:3]!='hps' and k[0:2]!='wh'}
#             # for param in mobilenet_state_dict:
#             #     param.requires_grad = False
#             mobilenet_state_dict.update(pretrained_dict)
#             self.load_state_dict(mobilenet_state_dict, strict=False)
#             a = 1
#
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#                 elif isinstance(m, nn.Linear):
#                     n = m.weight.size(1)
#                     m.weight.data.normal_(0, 0.01)
#                     m.bias.data.zero_()
#
# def get_pose_net(heads, head_conv,num_layers=1):
#     path ='/home/wuxilab/dxx/CenterNet/exp/multi_pose/pretrained_256deconv_focal_/model_best.pth'
#     model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
#     model._initialize_weights(path=path)
#     # for i in model.parameters():
#     #     i.requires_grad = False
#     # for i in model.wh.parameters():
#     #     i.requires_grad = True
#     # for i in model.hp_offset.parameters():
#     #     i.requires_grad = True
#     # for i in model.hps.parameters():
#     #     i.requires_grad = True
#     # for i in model.reg.parameters():
#     #     i.requires_grad = True
#     return model
#





# ## pretrained_16deconv_only for focal
#
# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Dequan Wang and Xingyi Zhou
# # ------------------------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import math
# import logging
#
# import torch
# import torch.nn as nn
# # from .DCNv2.dcn_v2 import DCN
# import torch.utils.model_zoo as model_zoo
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.normal_(m.weight, std=0.001)
#             # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             # torch.nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
#         self.inplanes = 1280
#         self.deconv_with_bias = False
#         self.heads = heads
#         super(MobileNetV2, self).__init__()
#
#         block = block
#         input_channel = 32
#         last_channel = 1280
#
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # used for deconv layers
#         llast = 16
#         self.deconv_layers = self._make_deconv_layer(
#             3,
#             [16, 16, llast ],
#             [4, 4, 4],
#         )
#         # self.final_layer = []
#         wh_hp = 64
#         for head in sorted(self.heads):
#             num_output = self.heads[head]
#             if head_conv > 0:
#
#
#                 if head == 'wh':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 elif head == 'hps':
#                     fc = nn.Sequential(
#                         nn.Conv2d(self.heads['hm']+self.heads['hm_hp'], wh_hp,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(wh_hp, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#                 else:
#                     fc = nn.Sequential(
#                         nn.Conv2d(llast, head_conv,
#                                   kernel_size=3, padding=1, bias=True),
#                         nn.ReLU(inplace=True),
#                         nn.Conv2d(head_conv, num_output,
#                                   kernel_size=1, stride=1, padding=0))
#             else:
#
#                 fc = nn.Conv2d(
#                     in_channels=llast,
#                     out_channels=num_output,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0
#                 )
#             self.__setattr__(head, fc)
#
#
#
#
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#
#         layers = []
#         for i in range(num_layers):
#             kernel, padding, output_padding = \
#                 self._get_deconv_cfg(num_kernels[i], i)
#
#             planes = num_filters[i]
#             layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=self.inplanes,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=self.deconv_with_bias))
#             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
#             layers.append(nn.ReLU(inplace=True))
#             self.inplanes = planes
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         # x = x.mean(3).mean(2)
#
#         x = self.deconv_layers(x)
#
#         hm_x = self.hm(x)
#         hm_hp_x = self.hm_hp(x)
#         # hm_hp_all = torch.cat([hm_hp_x, hm_x],1)
#         # wh = self.wh(hm_hp_all)
#         # hps = self.hps(hm_hp_all)
#         # hp_offset = self.hp_offset(x)
#         # reg = self.reg(x)
#
#
#         ret = {'hm':hm_x, 'hm_hp':hm_hp_x}
#
#
#
#
#         return [ret]
#
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             padding = 1
#             output_padding = 0
#
#
#         return deconv_kernel, padding, output_padding
#
#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()
#
#     def _initialize_weights(self,path):
#         if path!=None:
#             # print('=> init resnet deconv weights from normal distribution')
#             for _, m in self.deconv_layers.named_modules():
#                 if isinstance(m, nn.ConvTranspose2d):
#                     # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.normal_(m.weight, std=0.001)
#                     if self.deconv_with_bias:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     # print('=> init {}.weight as 1'.format(name))
#                     # print('=> init {}.bias as 0'.format(name))
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#             # print('=> init final conv weights from normal distribution')
#             for head in self.heads:
#                 final_layer = self.__getattr__(head)
#                 for i, m in enumerate(final_layer.modules()):
#                     if isinstance(m, nn.Conv2d):
#                         # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         # print('=> init {}.weight as normal(0, 0.001)'.format(name))
#                         # print('=> init {}.bias as 0'.format(name))
#                         if m.weight.shape[0] == self.heads[head]:
#                             if 'hm' in head:
#                                 nn.init.constant_(m.bias, -2.19)
#                             else:
#                                 nn.init.normal_(m.weight, std=0.001)
#                                 nn.init.constant_(m.bias, 0)
#             pretrained_dict = torch.load(path)
#             mobilenet_state_dict = self.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict}
#             mobilenet_state_dict.update(pretrained_dict)
#             self.load_state_dict(mobilenet_state_dict, strict=False)
#
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.zero_()
#                 elif isinstance(m, nn.Linear):
#                     n = m.weight.size(1)
#                     m.weight.data.normal_(0, 0.01)
#                     m.bias.data.zero_()
#
# def get_pose_net(heads, head_conv,num_layers=1):
#     path ='/home/wuxilab/dxx/CenterNet/src/mobilenet_v2.pth.tar'
#     model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
#     model._initialize_weights(path=path)
#
#     return model



#
## pretrained_16deconv_先focalh后全部

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
# from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

def _sigmoid(x):
  y = torch.clamp(torch.sigmoid(x)  , min=1e-4, max=1-1e-4)
  return y

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    def __init__(self, block, heads, head_conv, input_size=224, width_mult=1, path = None):
        self.inplanes = 1280
        self.deconv_with_bias = False
        self.heads = heads
        super(MobileNetV2, self).__init__()

        block = block
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # used for deconv layers
        llast = 16
        self.deconv_layers = self._make_deconv_layer(
            3,
            [16, 16, llast ],
            [4, 4, 4],
        )
        # self.final_layer = []
        wh_hp = 64
        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:


                if head == 'wh':
                    fc = nn.Sequential(
                        nn.Conv2d(llast+18, wh_hp,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(wh_hp, num_output,
                                  kernel_size=1, stride=1, padding=0))
                elif head == 'hps':
                    fc = nn.Sequential(
                        nn.Conv2d(llast+18, wh_hp,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(wh_hp, num_output,
                                  kernel_size=1, stride=1, padding=0))
                else:
                    fc = nn.Sequential(
                        nn.Conv2d(llast, head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output,
                                  kernel_size=1, stride=1, padding=0))
            else:

                fc = nn.Conv2d(
                    in_channels=llast,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)





    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)

        x = self.deconv_layers(x)

        hm_x = self.hm(x)
        hm_hp_x = self.hm_hp(x)
        hm_hp_all = torch.cat([hm_hp_x , hm_x, x],1)
        wh = self.wh(hm_hp_all)
        hps = self.hps(hm_hp_all)
        hp_offset = self.hp_offset(x)
        reg = self.reg(x)

        # ret = {'hm': hm_x, 'hm_hp': hm_hp_x, 'wh': wh, 'hps': hps, 'reg': reg,
        #                'hp_offset':hp_offset}

        ret = (hm_x, hm_hp_x, wh, hps, reg, hp_offset)

        return ret
        # return [ret]

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            padding = 1
            output_padding = 0


        return deconv_kernel, padding, output_padding

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()

    def _initialize_weights(self,path):
        if path!=None:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)
            pretrained_model = torch.load(path, map_location= lambda storage, loc: storage)
            pretrained_dict = pretrained_model['state_dict']
            mobilenet_state_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mobilenet_state_dict and k[0:3]!='hps' and k[0:2]!='wh'}
            # for param in mobilenet_state_dict:
            #     param.requires_grad = False
            mobilenet_state_dict.update(pretrained_dict)
            self.load_state_dict(mobilenet_state_dict, strict=False)
            a = 1

        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

def get_pose_net(heads, head_conv,num_layers=1):
    path ='/home/wuxilab/dxx/CenterNet/exp/multi_pose/pretrained_16deconv_only_focal/model_best.pth'
    model = MobileNetV2(InvertedResidual, heads, head_conv ,path =path )
    model._initialize_weights(path=path)
    # for i in model.parameters():
    #     i.requires_grad = False
    # for i in model.wh.parameters():
    #     i.requires_grad = True
    # for i in model.hp_offset.parameters():
    #     i.requires_grad = True
    # for i in model.hps.parameters():
    #     i.requires_grad = True
    # for i in model.reg.parameters():
    #     i.requires_grad = True
    return model


