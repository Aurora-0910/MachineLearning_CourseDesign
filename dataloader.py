import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

class LiverDataset(data.Dataset):
    # 类的初始化函数，根据指定的图片路径，读取所有图片数据
    def __init__(self, root, transform=None, target_transform=None, mode='train'):
        n = len(os.listdir(root + '/images'))
        imgs = []
        for i in range(n):
            img = os.path.join(root, 'images',"%d.png" % (i + 1))
            mask = os.path.join(root, 'mask', "%d.png" % (i + 1))
            imgs.append([img, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    # 数据获取函数，定义数据怎么读怎么处理，数据预处理、数据增强也都可以在这里进行
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    #  返回数据长度的函数
    def __len__(self):
        return len(self.imgs)
