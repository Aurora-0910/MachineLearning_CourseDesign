import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

from unet import Unet
from dataloader import LiverDataset

epochs = 100
batch_size = 32   # 32
device = 'cpu'  # 选择设备，有cuda用cuda，没有就用cpu
best_model = None
best_loss = 999  # best_loss统计，初始化
save_path = 'best_model.pkl'


x_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.Grayscale(num_output_channels=1)
])

y_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224)
])

# 加载训练集
liver_dataset = LiverDataset("./data/train", transform=x_transform, target_transform=y_transform)
dataloader = torch.utils.data.DataLoader(liver_dataset, batch_size=batch_size, shuffle=True)

# 加载模型
model = Unet()
# 定义Loss算法
criterion = torch.nn.BCELoss()      # 就是一个将sigmoid函数和BCELOSS函数结合的一种loss函数
# 定义RMSprop算法
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# 训练epochs次
for epoch in range(epochs):
    model.train()   # 打开训练模式
    epoch_loss = 0
    train_bar = tqdm(dataloader)
    # 按照batch_size开始训练
    for x, y in train_bar:
        print(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        optimizer.zero_grad()
        # 将数据拷贝到device中
        inputs = x.to(device)
        labels = y.to(device)
        # 使用网络参数，输出预测结果
        outputs = model(inputs)
        # print(outputs)
        # 计算loss
        loss = criterion(outputs, labels)
        # 更新参数
        loss.backward()  #  反向传播
        optimizer.step()
        epoch_loss += loss.item()

    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % str(epoch_loss))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = model.state_dict()

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存训练集 loss 值最低的网络参数作为最佳模型参数。
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
