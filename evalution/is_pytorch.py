'''
Descripttion: 
version: 
Author: Yukai
Date: 2022-12-13 21:33:12
LastEditors: Yukai
LastEditTime: 2023-03-27 13:05:31
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from scipy.stats import entropy
from torchvision.models.inception import inception_v3


import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ISImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root) + "/*.png"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')      
        item_image = self.transform(img)
        return item_image

    def __len__(self):
        return len(self.files)


import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "7"  #（代表仅使用第0，1号GPU）
# path = "/data3/yukai/text2image/testssd/netG_120/gen/"
path ='/data3/yukai/text2image/tmp/netG_400/gen'
count = 0
for root,dirs,files in os.walk(path):    #遍历统计
      for each in files:
             count += 1   #统计文件夹下文件个数
print(count)
batch_size = 128
transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

val_dataloader = DataLoader(
    ISImageDataset(path, transforms_=transforms_),
    batch_size = batch_size,
)

cuda = True if torch.cuda.is_available() else False
# cuda = cuda + ":1"
print('cuda: ',cuda)
tensor = torch.cuda.FloatTensor

inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model.eval()
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).cuda()

def get_pred(x):
    if True:
        x = up(x)
    x = inception_model(x)
    return F.softmax(x, dim=1).data.cpu().numpy()

print('Computing predictions using inception v3 model')
preds = np.zeros((count, 1000))

for i, data in enumerate(val_dataloader):
    data = data.type(tensor)
    batch_size_i = data.size()[0]
    preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(data)

print('Computing KL Divergence')
split_scores = []
splits=10
N = count
for k in range(splits):
    part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
    py = np.mean(part, axis=0)  # marginal probability
    # print(py)
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]  # conditional probability
        scores.append(entropy(pyx, py))  # compute divergence
        # print(scores)
    split_scores.append(np.exp(np.mean(scores)))


mean, std  = np.mean(split_scores), np.std(split_scores)
print('IS is %.4f' % mean)
print('The std is %.4f' % std)
