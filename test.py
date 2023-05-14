import numpy as np
import torch
from torch import nn
from PIL import Image


from model.lite_mono import LiteMono

model = LiteMono().cuda()

print(model)

img = Image.open("image/0000000000.png").resize( (640, 192), Image.BILINEAR ).convert('RGB')

img = np.array(img, dtype=np.float32)
img = torch.tensor(img).cuda()
print(img.shape)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).permute((0, 3, 1, 2))
print(img.shape, '\n\n')


# For test forward propogation
for i in range(100):
    y = model(img)#output: [1/4 depth map, 1/2 depth map, full res depth map]


# For test backward propogation and optimize (fake loss)
loss = y[0].sum() + y[1].sum() + y[2].sum() 
loss.backward()
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

print(y[0].shape)
print(y[1].shape)
print(y[2].shape)