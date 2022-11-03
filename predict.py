import glob

import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

from unet import Unet
from dataloader import LiverDataset

batch_size = 32   # 32
device = 'cpu'


x_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
])

y_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224)
])


# 加载网络
model = Unet()
# 将网络拷贝到deivce中
model.to(device=device)
# 加载模型参数
model.load_state_dict(torch.load('./best_model.pkl', map_location=device))
# 测试模式
model.eval()
# 读取所有图片路径
tests_path = glob.glob('data/test/Images/*.png')
print(tests_path)
# 遍历素有图片
i = 1
for test_path in tests_path:
    # 保存结果地址
    # save_res_path = test_path.split('/')[6] + '_res.png'
    save_res_path = 'data/result/'+str(i) + '_res.png'
    i = i+1
    print(save_res_path)
    # 读取图片
    img = cv2.imread(test_path)
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.dtype)
    print(img.shape[0],img.shape[1])
    # 转为batch为1，通道为1，大小为512*512的数组
    # img=img[:128,:128]
    # print(img.shape)
    # img = img.reshape(1, 1, 128, 128)
    # print(img.shape)

    # 转为tensor
    img_tensor = np.array(img)
    img_tensor = x_transform(img_tensor)
    img_tensor=img_tensor.reshape(1,1,224,224)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = model(img_tensor)
    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
    # 处理结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    # 保存图片
    cv2.imwrite(save_res_path, pred)




