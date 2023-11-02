import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from torchvision import models
import cv2
from pyefun import *

from hanzi import mobilenet_v3 as net

max_label_len = 4
label_map = []


# 定义验证码识别模型类
class Model_Captcha:
    # 定义类属性
    IMAGE_SHAPE = (60, 160)
    transform = T.Compose([
        T.ToPILImage(),  # 将图像转为PIL图像
        T.Resize(IMAGE_SHAPE),  # 调整图像大小
        T.ToTensor(),  # 将图像转为Tensor
    ])

    def __init__(self, model_path):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = net.Net(num_classes=len(label_map) * max_label_len)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.DEVICE)
        self.model.eval()

    def predict(self, im):
        im = self.transform(im)
        im = im.to(self.DEVICE)
        im = im.unsqueeze(0)
        out = self.model(im)
        out = out.view(-1, max_label_len, len(label_map))
        predict = torch.argmax(out, dim=2)
        label = predict.cpu().detach().numpy().tolist()[0]
        return ''.join(label_map[i] for i in label)


# 加载标签文件
def load_label_file(labelFile):
    data_map = 分割文本(读入文本(labelFile), "\r\n")
    # 删除空行
    data_map = [i for i in data_map if i != ""]
    return data_map


if __name__ == '__main__':
    label_file = r'./label.txt'
    model_path = "./models/save_4.model"
    test_path = "./test"

    label_map = load_label_file(label_file)
    print("加载分类数量", len(label_map))
    model = Model_Captcha(model_path)

    文件列表 = 文件_枚举(test_path, ".png", False)
    正确率 = 0
    识别成功 = 0
    识别失败 = 0
    文件数量 = len(文件列表)

    for 文件路径 in 文件列表:
        文件名 = 文件_取文件名(文件路径)
        正确结果 = strCut(文件名, "$_")
        图片数据 = 读入文件(文件路径)
        im = cv2.imdecode(np.frombuffer(图片数据, np.uint8), cv2.IMREAD_COLOR)
        识别结果 = model.predict(im)

        if 正确结果 == 识别结果:
            识别成功 += 1
            print("文件名", 文件名, "正确结果", 正确结果, "识别结果", 识别结果, "正确")
        else:
            识别失败 += 1
            print("文件名", 文件名, "正确结果", 正确结果, "识别结果", 识别结果, "错误")

    正确率 = 识别成功 / 文件数量 * 100
    print("正确率", 正确率, "文件数量", 文件数量, "识别成功", 识别成功, "识别失败", 识别失败)
