import torch
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

class Net(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(Net, self).__init__()
        if pretrained:
            self.net = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.net = models.mobilenet_v3_large()
        print(self.net)
        # 修改最后层 输出的分类数量
        in_features = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        # 前向传播函数，将输入x传递给ResNet模型
        x = self.net(x)
        return x
