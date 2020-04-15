from models.model import create_model, load_model
import sys
import torch.nn as nn
import torch
import numpy as np
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
heads = {'hm': 1, 'wh': 2, 'hps': 34,'reg': 2,'hm_hp': 17,'hp_offset': 2}

model = create_model('res_18', heads, head_conv=64)
# model = load_model(model, model_path='/home/wuxilab/dxx/CenterNet/exp/multi_pose/res_18_planer/model_best.pth')

# x = torch.tensor(np.random.random((1,3,512,512)).astype('float32'))
# torch.onnx.export(model, x, 'tensorrt.onnx', verbose=True,
#         input_names=None, output_names=None)

model = model.cuda()
model.eval()
inputt = torch.Tensor(1,3,512,512).cuda()
model(inputt)
with torch.no_grad():
    s = time.time()
    num = 10
    for i in range(num):
        model(inputt)
print((time.time()-s)/num)
#




# import planer
# from planer import torch2planer, read_net
#
# import cupy as cp
#
# torch2planer(model, 'res18', x= torch.Tensor(1,3,512,512))
# pal = planer.core(cp)
#
#
# x = cp.asarray(cp.random.random((1,3,512,512))).astype('float32')
# net = read_net('res18')
# net(x)
# # net.show()
# # print(y.shape)

