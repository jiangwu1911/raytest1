# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 增强的数据预处理和数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 使用预训练的ResNet-18模型，并修改全连接层以适配CIFAR-10的10个类别
def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(pretrained=True)
    # 修改最后的全连接层，以适应CIFAR-10的10个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18(num_classes=10)
model = model.to(device)

# 使用交叉熵损失和Adam优化器，并添加学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch将学习率乘以0.1

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
    return test_loss, test_acc

# 训练和测试循环
num_epochs = 50
print("开始训练...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader, criterion)
    scheduler.step()  # 更新学习率
    print('-' * 50)

end_time = time.time()
print(f'训练完成，耗时: {(end_time - start_time) // 60:.0f}分 {(end_time - start_time) % 60:.0f}秒')
