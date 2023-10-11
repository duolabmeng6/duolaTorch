import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights

# 定义自定义神经网络模型
class Net(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(Net, self).__init__()
        if pretrained:
            # 使用预训练的ResNet-34模型
            self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            # 创建一个新的ResNet-34模型
            self.resnet = models.resnet34()
        in_features = self.resnet.fc.in_features
        # 替换ResNet模型的全连接层，将输出维度设置为num_classes
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 前向传播函数，将输入x传递给ResNet模型
        x = self.resnet(x)
        return x
