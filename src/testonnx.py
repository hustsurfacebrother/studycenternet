from models.model import create_model, load_model
import sys
import torch.nn as nn
import torch
import numpy as np

heads = {'hm': 1, 'wh': 2, 'hps': 34,'reg': 2,'hm_hp': 17,'hp_offset': 2}

model = create_model('res_18', heads, head_conv=64)
model = load_model(model, model_path='/home/wuxilab/dxx/CenterNet/exp/multi_pose/res_18_upsample/model_best.pth')

model.eval()
torch.onnx.export(model,
                  torch.Tensor(1,3,512,512),
                  '/home/wuxilab/dxx/CenterNet/out.onnx',
                  verbose=True,
                    )



#
import planer
from planer import read_onnx
import numpy as np
#
# net = read_onnx('out')
# # input should be float32
#
#
# pal = planer.core(np)
# # the same folder should contain resnet18.txt, resnet18.npy
# x = pal.random.randn(1, 3, 512, 512).astype('float32')
# y = net(x)
#
# net.show()
# print(y.shape)
