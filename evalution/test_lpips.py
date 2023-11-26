'''
Descripttion: 
version: 
Author: Yukai
Date: 2022-12-03 14:10:56
LastEditors: Yukai
LastEditTime: 2022-12-19 17:15:48
'''
'''test diversity'''
import torch
import numpy as np
import lpips
from PIL import Image
import os
import torchvision.transforms as T
import argparse
from tqdm import tqdm
import pickle

def main(args):
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = [T.ToTensor()]
    transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    # transform.append(T.Resize(256,256))
    im_transform = T.Compose(transform)

    orig_images = os.listdir(args.orig_image_path)
    data = util("/home/yukai/exp/txt2im/data/birds/test/filenames.pickle")
    # N = len(orig_images)
    N = 2900
    print(N)
    net = lpips.LPIPS(net='alex')
    net = net.cuda()
    net.eval()
    scores = []
    with torch.no_grad():

        for i in range(len(data)):
            orig_image = im_transform((Image.open(os.path.join(args.orig_image_path, data[i] + ".jpg")).convert('RGB')).resize((256,256),Image.NEAREST))
            orig_image = orig_image.cuda()
            orig_image = orig_image.unsqueeze(0)
            for j in range(args.generated_image_number):
                generated_image = im_transform(Image.open(os.path.join(args.generated_image_path, 'img_s' + str(j) + '.png')).convert('RGB'))
                generated_image = generated_image.cuda()
                generated_image = generated_image.unsqueeze(0)
                score = net(orig_image, generated_image).squeeze()
                scores.append(score.cpu().numpy())
    scores_all = np.asarray(scores)
    scores_mean = np.mean(scores_all)
    scores_std = np.std(scores_all)
    print('mean diversity scores = %4.2f%% +- %4.2f%%' % (scores_mean, scores_std))
def util(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        # print(len(data))
        # print(data)
    return data


def saveimg():
    data = util("/home/yukai/exp/txt2im/data/coco/test/filenames.pickle")
    
    for i in range(len(data)):
        print(i)
        im  = Image.open(os.path.join(args.orig_image_path, data[i] + ".jpg")).convert('RGB').resize((256,256),Image.NEAREST)
        
        im.save("/home/yukai/exp/txt2im/data/coco/test/image/" + str(i) + ".png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_image_path', type=str, default="/home/yukai/exp/txt2im/data/coco/val2014")
    parser.add_argument('--generated_image_path', type=str, default="/home/yukai/exp/txt2im/tmp/bird_myexp02/64/models/netG_180/gen")
    parser.add_argument('--generated_image_number', type=int, default= 2900)
    args = parser.parse_args()
    # util("/home/yukai/exp/txt2im/data/birds/test/filenames.pickle")
    saveimg()
    # main(args)
