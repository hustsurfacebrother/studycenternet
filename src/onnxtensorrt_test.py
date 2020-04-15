import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torch
import time
model = onnx.load("tensorrt.onnx")
engine = backend.prepare(model, device='CUDA:7')
input_data = np.random.random(size=(1, 3, 512, 512)).astype(np.float32)
output_data = engine.run(input_data)
output_data = engine.run(input_data)
output_data = engine.run(input_data)

# model = model.cuda()
# model.eval()
# inputt = torch.Tensor(1,3,512,512).cuda()
# model(inputt)
# with torch.no_grad():
#     s = time.time()
#     num = 10
#     for i in range(num):
#         model(inputt)
# print((time.time()-s)/num)

start = time.time()
num = 10
for i in range(num):
    output_data = engine.run(input_data)
print((time.time()-start)/num)


# print(output_data[0].shape)
# print(output_data[1].shape)
# print(output_data[2].shape)
# print(output_data[3].shape)
# print(output_data[4].shape)
# print(output_data[5].shape)