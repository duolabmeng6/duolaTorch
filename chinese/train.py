import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from pyefun import *

# from hanzi import resnet34
from hanzi import mobilenet_v3 as net

# 定义一些超参数
image_shape = (60, 160)
label_map = []  # 标签映射，用于将字符标签转为数字
max_label_len = 4  # 最大标签长度

# 数据预处理和增强
transform = T.Compose([
    T.ToPILImage(),  # 转为PIL图像
    T.Resize(image_shape),  # 调整图像大小
    T.ToTensor(),  # 转为Tensor
])

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_path, label_map, max_label_len):
        super(MyDataset, self).__init__()
        # 获取数据文件列表并提取标签
        self.data = [(os.path.join(data_path, file), file.split('.')[0]) for file in os.listdir(data_path)]
        self.label_map = [char for char in label_map]
        self.label_map_len = len(self.label_map)
        self.max_label_len = max_label_len

    def __getitem__(self, index):
        file = self.data[index][0]
        label = self.data[index][1]
        label = strCut(label, "$_")  # 删除标签中的"$_"字符
        raw_len = len(label)
        im = np.fromfile(file, dtype=np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        im = transform(im)  # 图像转换和增强
        label = [self.label_map.index(i) for i in label]  # 将字符标签转为数字
        label = torch.as_tensor(label, dtype=torch.int64)
        label = F.one_hot(label, num_classes=len(label_map)).float()  # 使用one-hot编码标签
        return im, label, raw_len

    def __len__(self):
        return len(self.data)

# 封装的训练函数
def train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, stop_accuracy=None, model_save_dir="models",
                model_save_interval=1):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查并选择GPU或CPU
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, label, _ in tqdm(train_loader, 'Training'):
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)
            label = label.view(-1, max_label_len * len(label_map))
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples

        # 验证阶段
        model.eval()
        total_loss = 0.0
        correct = count = 0

        for x, label, _ in tqdm(test_loader, 'Validation'):
            x, label = x.to(DEVICE), label.to(DEVICE)
            out = model(x)
            label_copy = label.view(-1, max_label_len * len(label_map))
            loss = loss_func(out, label_copy)

            out = out.view(-1, max_label_len, len(label_map))
            predict = torch.argmax(out, dim=2)
            label = torch.argmax(label, dim=2)

            count += x.size(0) * max_label_len
            correct += (predict == label).sum().item()

            total_loss += loss.item()

        validation_loss = total_loss / count
        accuracy = correct / count

        lr = optimizer.param_groups[0]['lr']

        print(
            f"\r\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {lr}")
        scheduler.step()

        # 保存模型
        save = False
        if (epoch + 1) % model_save_interval == 0:
            if not 文件是否存在(model_save_dir):
                创建目录(model_save_dir)
            save = True
            model_save_path = os.path.join(model_save_dir, f"save_{epoch}.model")
            torch.save(model.state_dict(), model_save_path)

        if stop_accuracy and accuracy >= stop_accuracy:
            print("\r\nAccuracy is over!")
            if not save:
                model_save_path = os.path.join(model_save_dir, f"save_{epoch}.model")
                torch.save(model.state_dict(), model_save_path)
            break

# 加载标签文件
def load_label_file(labelFile):
    data_map = 分割文本(读入文本(labelFile), "\r\n")
    # 删除空行
    data_map = [i for i in data_map if i != ""]
    return data_map

if __name__ == '__main__':
    label_file = r'./label.txt'
    label_map = load_label_file(label_file)
    print("加载分类数量", len(label_map))

    train_loader = DataLoader(
        dataset=MyDataset(r'./train', label_map=label_map, max_label_len=max_label_len),
        batch_size=32, shuffle=True,
        num_workers=0)

    test_loader = DataLoader(
        dataset=MyDataset(r'./test', label_map=label_map, max_label_len=max_label_len),
        batch_size=4, shuffle=True,
        num_workers=0)

    model = net.Net(num_classes=len(label_map) * max_label_len, pretrained=True)
    # 加载自己的模型
    # model.load_state_dict(torch.load(r'./models/save_1.model'))

    train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, stop_accuracy=1, model_save_dir="models",
                model_save_interval=10)
