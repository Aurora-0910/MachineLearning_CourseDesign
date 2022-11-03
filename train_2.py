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

epochs = 100
batch_size = 128   #32
device = 'cpu'
best_model = None
best_loss = 999
save_path = 'best_model.pkl'

"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(1, 32)
        self.D1 = DownSampling(32)
        self.C2 = Conv(32, 64)
        self.D2 = DownSampling(64)
        self.C3 = Conv(64, 128)
        self.D3 = DownSampling(128)
        self.C4 = Conv(128, 256)
        self.D4 = DownSampling(256)
        self.C5 = Conv(256, 512)

        # 4次上采样
        self.U1 = UpSampling(512)
        self.C6 = Conv(512, 256)
        self.U2 = UpSampling(256)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 1, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode='train'):
        n = len(os.listdir(root + '/images'))

        imgs = []

        if mode == 'train':
            for i in range(n):
                img = os.path.join(root, 'images',
                                   "%d.png" % (i + 1))
                mask = os.path.join(root, 'mask', "%d.png" % (i + 1))
                imgs.append([img, mask])
        else:
            for i in range(n):
                img = os.path.join(root, 'images',
                                   "%d.png" % (i + 1))
                mask = os.path.join(root, 'mask', "%d.png" % (i + 1))
                imgs.append([img, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


x_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.Grayscale(num_output_channels=1)
])

y_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224)
])

liver_dataset = LiverDataset("./data/train", transform=x_transform, target_transform=y_transform)
dataloader = torch.utils.data.DataLoader(liver_dataset, batch_size=batch_size, shuffle=True)

model = UNet()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.2)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        optimizer.zero_grad()
        inputs = x.to(device)
        labels = y.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % str(epoch_loss))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = model.state_dict()

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, save_path)

print('Finished Training')

plt.figure('测试一张图片')
pil_img = Image.open('./data/train/images/1.png')
np_img = np.array(pil_img)
plt.imshow(np_img)
plt.show()

plt.figure('测试一张蒙版图片')
pil_img = Image.open('./data/train/mask/1.png')
np_img = np.array(pil_img)
plt.imshow(np_img)
plt.show()
