import torch
from torch import nn
import torchvision.transforms as T
from torchvision import models
import cv2
from pyefun import *
import cv2
from pyefun import 文件_枚举, 文件_取文件名, strCut

Max_label_len = 4
LABEL_MAP = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet18 = models.resnet18(num_classes=Max_label_len * len(LABEL_MAP))

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Model_Captcha:
    # Define class attributes
    IMAGE_SHAPE = (60, 160)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SHAPE),
        T.ToTensor(),
    ])

    def __init__(self, model_path):

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.DEVICE)
        self.model.eval()

    def predict(self, im):
        im = self.transform(im)
        im = im.to(self.DEVICE)
        im = im.unsqueeze(0)
        out = self.model(im)
        out = out.view(-1, Max_label_len, len(LABEL_MAP))
        predict = torch.argmax(out, dim=2)
        label = predict.cpu().detach().numpy().tolist()[0]
        return ''.join(LABEL_MAP[i] for i in label)


def load_label_file(labelFile):
    data_map = 分割文本(读入文本(labelFile), "\r\n")
    # 把 data_map 空行删除
    data_map = [i for i in data_map if i != ""]
    # print(LABEL_MAP)
    return data_map


if __name__ == '__main__':
    label_file = r'./label.txt'
    model_path = "./models/save_2.model"
    test_path = "./test"

    LABEL_MAP = load_label_file(label_file)
    print("加载分类数量", len(LABEL_MAP))
    model = Model_Captcha(model_path)

    文件列表 = 文件_枚举(test_path, ".png", False)
    正确率 = 0
    识别成功 = 0
    识别失败 = 0
    文件数量 = len(文件列表)
    for 文件路径 in 文件列表:
        文件名 = 文件_取文件名(文件路径)
        正确结果 = strCut(文件名, "$_")
        im = cv2.imread(文件路径, cv2.IMREAD_COLOR)
        识别结果 = model.predict(im)
        if 正确结果 == 识别结果:
            识别成功 += 1
            print("文件名", 文件名, "正确结果", 正确结果, "识别结果", 识别结果, "正确")
        else:
            识别失败 += 1
            print("文件名", 文件名, "正确结果", 正确结果, "识别结果", 识别结果, "错误")

    正确率 = 识别成功 / 文件数量 * 100
    print("正确率", 正确率, "文件数量", 文件数量, "识别成功", 识别成功, "识别失败", 识别失败)
