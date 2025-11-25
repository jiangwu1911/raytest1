# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

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

# 使用预训练的ResNet-18模型
def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18(num_classes=10)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

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

# 生成混淆矩阵
def plot_confusion_matrix(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 创建混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    return cm, all_labels, all_preds

# 生成ROC曲线
def plot_roc_curve(model, test_loader, classes):
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # One-hot encode labels
    one_hot_labels = np.eye(len(classes))[all_labels]
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ROC curve saved as 'roc_curve.png'")
    
    # 计算宏观平均AUC
    macro_auc = roc_auc_score(one_hot_labels, all_outputs, multi_class='ovr', average='macro')
    print(f"📊 Macro-average ROC-AUC: {macro_auc:.4f}")
    
    return macro_auc

# 计算详细分类指标
def calculate_detailed_metrics(all_labels, all_preds, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(classes))
    )
    
    # 打印每个类别的指标
    print("\n📈 Detailed Classification Report:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for i, class_name in enumerate(classes):
        print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # 加权平均指标
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    print("-" * 60)
    print(f"{'Weighted':<10} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f}")
    
    return weighted_precision, weighted_recall, weighted_f1

# 主训练循环
num_epochs = 50
print("开始训练...")
start_time = time.time()

best_acc = 0
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader, criterion)
    scheduler.step()
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print('-' * 50)

end_time = time.time()
print(f'训练完成，耗时: {(end_time - start_time) // 60:.0f}分 {(end_time - start_time) % 60:.0f}秒')
print(f'最佳测试准确率: {best_acc:.2f}%')

# 加载最佳模型进行最终评估
print("\n🔍 加载最佳模型进行最终评估...")
model.load_state_dict(torch.load('best_model.pth'))

# 生成混淆矩阵
print("\n📊 生成混淆矩阵...")
cm, all_labels, all_preds = plot_confusion_matrix(model, test_loader, classes)

# 生成ROC曲线
print("\n📈 生成ROC曲线...")
macro_auc = plot_roc_curve(model, test_loader, classes)

# 计算详细指标
print("\n🧮 计算详细分类指标...")
weighted_precision, weighted_recall, weighted_f1 = calculate_detailed_metrics(all_labels, all_preds, classes)

# 打印总体准确率
final_accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\n🎯 最终测试准确率: {final_accuracy:.2f}%")
print(f"📊 加权精确率: {weighted_precision:.4f}")
print(f"📊 加权召回率: {weighted_recall:.4f}")
print(f"📊 加权F1分数: {weighted_f1:.4f}")
print(f"📊 宏观平均AUC: {macro_auc:.4f}")

print("\n✅ 所有图表已生成完成!")
print("   - confusion_matrix.png: 混淆矩阵")
print("   - roc_curve.png: ROC曲线")
print("   - best_model.pth: 最佳模型权重")
