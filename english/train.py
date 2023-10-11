import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import torchvision.transforms as T
from pyefun import *

# 定义一些超参数
IMAGE_SHAPE = (60, 160)
# LABEL_MAP = [i for i in 到大写('0123456789abcdefghijklmnopqrstuvwxyz')]
Max_label_len = 4

# 数据预处理和增强
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SHAPE),
    T.ToTensor(),
])


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_path, label_map, max_label_len):
        super(MyDataset, self).__init__()
        self.data = [(os.path.join(data_path, file), file.split('.')[0]) for file in os.listdir(data_path)]
        self.label_map = [char for char in label_map]
        self.label_map_len = len(self.label_map)
        self.max_label_len = max_label_len

    def __getitem__(self, index):
        file = self.data[index][0]
        label = self.data[index][1]
        label = strCut(label, "$_")
        raw_len = len(label)
        im = np.fromfile(file, dtype=np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = transform(im)
        label = [self.label_map.index(i) for i in label]
        label = torch.as_tensor(label, dtype=torch.int64)
        label = F.one_hot(label, num_classes=len(LABEL_MAP)).float()
        return im, label, raw_len

    def __len__(self):
        return len(self.data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用默认的预训练模型
        # self.resnet = models.resnet(num_classes=Max_label_len * len(LABEL_MAP)) 等于下面两行
        in_features = self.resnet.fc.in_features # 获取最后一层的输入特征数
        self.resnet.fc = nn.Linear(in_features, Max_label_len * len(LABEL_MAP)) # 修改最后一层的输出特征数


    def forward(self, x):
        x = self.resnet(x)
        return x


# 封装的训练函数
def train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, stop_accuracy=None, model_save_dir="models",
                model_save_interval=1):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_func = nn.MSELoss() # 均方误差
    loss_func = nn.CrossEntropyLoss() # 交叉熵损失函数
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, label, _ in tqdm(train_loader, 'Training'):
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)
            label = label.view(-1, Max_label_len * len(LABEL_MAP))
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples

        # Validation
        model.eval()
        total_loss = 0.0
        correct = count = 0

        for x, label, _ in tqdm(test_loader, 'Validation'):
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)
            label_copy = label.view(-1, Max_label_len * len(LABEL_MAP))
            loss = loss_func(out, label_copy)

            out = out.view(-1, Max_label_len, len(LABEL_MAP))
            predict = torch.argmax(out, dim=2)
            label = torch.argmax(label, dim=2)

            count += x.size(0) * Max_label_len
            correct += (predict == label).sum().item()

            total_loss += loss.item()

        validation_loss = total_loss / count
        accuracy = correct / count

        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {lr}")
        scheduler.step()

        # Save the model
        if (epoch + 1) % model_save_interval == 0:
            if not 文件是否存在(model_save_dir):
                创建目录(model_save_dir)
            model_save_path = os.path.join(model_save_dir, f"save_{epoch}.model")
            torch.save(model.state_dict(), model_save_path)

        if stop_accuracy and accuracy >= stop_accuracy:
            print("Accuracy is over 95%!")
            break


def load_label_file(labelFile):
    data_map = 分割文本(读入文本(labelFile), "\r\n")
    # 把 data_map 空行删除
    data_map = [i for i in data_map if i != ""]
    # print(LABEL_MAP)
    return data_map


if __name__ == '__main__':
    label_file = r'./label.txt'
    LABEL_MAP = load_label_file(label_file)
    print("加载分类数量", len(LABEL_MAP))

    train_loader = DataLoader(
        dataset=MyDataset(r'./train', label_map=LABEL_MAP, max_label_len=Max_label_len),
        batch_size=32, shuffle=True,
        num_workers=0)

    test_loader = DataLoader(
        dataset=MyDataset(r'./test', label_map=LABEL_MAP, max_label_len=Max_label_len),
        batch_size=4, shuffle=True,
        num_workers=0)

    model = Net()
    # 加载自己的模型
    # model.load_state_dict(torch.load(r'./models/save_1.model'))

    train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, stop_accuracy=0.95, model_save_dir="models",
                model_save_interval=1)
