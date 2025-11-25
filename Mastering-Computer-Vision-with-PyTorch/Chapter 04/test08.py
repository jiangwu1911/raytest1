# -*- coding: utf-8 -*-
import ray
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import warnings
import os
warnings.filterwarnings("ignore")

# 初始化 Ray
ray.init(f"ray://192.168.1.217:10001")  # 自动连接到 Ray 集群

print("Ray 集群信息:")
print(f"可用节点: {ray.available_resources()}")

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
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

# 定义模型类 - 修复 torchvision 警告
def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 使用 Ray 的远程函数进行数据加载和预处理
@ray.remote
def load_datasets():
    """在远程节点上加载数据集"""
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    return train_set, test_set

# 使用 Ray 的 Actor 类进行分布式训练
@ray.remote(num_gpus=0.5)  # 每个 Actor 使用 1 个 GPU
class TrainingWorker:
    def __init__(self, worker_id, num_workers):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = get_resnet18(num_classes=10)
        self.model = self.model.to(self.device)
        
        # 优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        print(f"Worker {worker_id} 初始化完成，使用设备: {self.device}")

    def train_epoch(self, data_shard, epoch):
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 创建数据加载器
        train_loader = DataLoader(data_shard, batch_size=128, shuffle=True)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 学习率调度
        self.scheduler.step()
        
        return {
            'worker_id': self.worker_id,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def get_model_weights(self):
        """获取模型权重"""
        return self.model.state_dict()

    def set_model_weights(self, weights):
        """设置模型权重"""
        self.model.load_state_dict(weights)

    def validate(self, test_loader):
        """在测试集上验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        test_loss = running_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'all_outputs': all_outputs
        }

# 修复的联邦平均算法
def federated_averaging(worker_weights):
    """对多个worker的权重进行平均"""
    averaged_weights = {}
    
    # 首先收集所有权重的键
    all_keys = set()
    for weights in worker_weights:
        all_keys.update(weights.keys())
    
    # 对每个键进行平均
    for key in all_keys:
        # 检查第一个worker的该键的数据类型
        first_weight = worker_weights[0][key]
        
        if first_weight.dtype in [torch.int64, torch.int32, torch.long]:
            # 对于整数类型的权重，我们直接复制第一个worker的值
            # 因为这些通常是buffer（如num_batches_tracked），不需要平均
            averaged_weights[key] = first_weight.clone()
        else:
            # 对于浮点数类型的权重，进行平均
            averaged_weights[key] = torch.zeros_like(first_weight)
            for weights in worker_weights:
                averaged_weights[key] += weights[key]
            averaged_weights[key] /= len(worker_weights)
    
    return averaged_weights

# 简化的权重平均（只平均可训练参数）
def simple_weight_average(worker_weights):
    """简化的权重平均，只处理可训练参数"""
    averaged_weights = {}
    
    # 只处理第一个worker的权重
    for key in worker_weights[0].keys():
        weight = worker_weights[0][key]
        
        # 跳过整数类型的buffer
        if weight.dtype in [torch.int64, torch.int32, torch.long]:
            averaged_weights[key] = weight.clone()
            continue
            
        # 对浮点数权重进行平均
        averaged_weights[key] = torch.zeros_like(weight)
        for weights in worker_weights:
            averaged_weights[key] += weights[key]
        averaged_weights[key] /= len(worker_weights)
    
    return averaged_weights

# 可视化函数
def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Distributed Training', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ray_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved as 'ray_confusion_matrix.png'")

def plot_roc_curve(all_labels, all_outputs, classes):
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    one_hot_labels = np.eye(len(classes))[all_labels]
    
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
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves - Distributed Training', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ray_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ROC curve saved as 'ray_roc_curve.png'")
    
    macro_auc = roc_auc_score(one_hot_labels, all_outputs, multi_class='ovr', average='macro')
    return macro_auc

# 主函数
def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("🚀 开始分布式训练...")
    
    # 加载数据集
    print("📦 加载数据集...")
    train_set, test_set = ray.get(load_datasets.remote())
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    
    # 配置训练参数
    num_workers = 2  # 可以根据集群GPU数量调整
    num_epochs = 50
    data_per_worker = len(train_set) // num_workers
    
    print(f"使用 {num_workers} 个 workers")
    print(f"每个 worker 处理 {data_per_worker} 个样本")
    
    # 创建 workers
    print("👥 创建训练 workers...")
    workers = [TrainingWorker.remote(i, num_workers) for i in range(num_workers)]
    
    # 数据分片
    data_shards = []
    for i in range(num_workers):
        start_idx = i * data_per_worker
        end_idx = start_idx + data_per_worker if i < num_workers - 1 else len(train_set)
        data_shard = torch.utils.data.Subset(train_set, range(start_idx, end_idx))
        data_shards.append(data_shard)
    
    # 训练循环
    best_acc = 0
    training_history = []
    
    print("🎯 开始训练循环...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")
        
        # 并行训练所有 workers
        futures = []
        for i, worker in enumerate(workers):
            future = worker.train_epoch.remote(data_shards[i], epoch)
            futures.append(future)
        
        # 收集训练结果
        results = ray.get(futures)
        
        # 打印每个 worker 的结果
        for result in results:
            print(f"Worker {result['worker_id']}: Loss={result['train_loss']:.3f}, Acc={result['train_acc']:.2f}%, LR={result['learning_rate']:.6f}")
        
        # 联邦平均：聚合模型权重（每5个epoch聚合一次）
        if epoch % 5 == 0 or epoch == num_epochs:
            print("🔄 聚合模型权重...")
            weight_futures = [worker.get_model_weights.remote() for worker in workers]
            worker_weights = ray.get(weight_futures)
            
            # 使用修复的权重平均函数
            averaged_weights = simple_weight_average(worker_weights)
            
            # 分发平均权重给所有 workers
            set_weight_futures = [worker.set_model_weights.remote(averaged_weights) for worker in workers]
            ray.get(set_weight_futures)
            print("✅ 模型权重聚合完成")
        
        # 验证（使用第一个worker）
        if epoch % 10 == 0 or epoch == num_epochs:
            print("🧪 模型验证...")
            validation_result = ray.get(workers[0].validate.remote(test_loader))
            test_acc = validation_result['test_acc']
            
            print(f"📊 验证准确率: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                # 保存最佳模型
                best_weights = ray.get(workers[0].get_model_weights.remote())
                torch.save(best_weights, 'ray_best_model.pth')
                print(f"🎉 新的最佳模型已保存，准确率: {best_acc:.2f}%")
            
            training_history.append({
                'epoch': epoch,
                'test_acc': test_acc,
                'worker_results': results
            })
    
    end_time = time.time()
    print(f'\n✅ 训练完成，耗时: {(end_time - start_time) // 60:.0f}分 {(end_time - start_time) % 60:.0f}秒')
    print(f'🎯 最佳测试准确率: {best_acc:.2f}%')
    
    # 最终评估
    print("\n🔍 最终评估...")
    
    # 加载最佳模型
    best_weights = torch.load('ray_best_model.pth')
    ray.get(workers[0].set_model_weights.remote(best_weights))
    
    # 最终验证
    final_result = ray.get(workers[0].validate.remote(test_loader))
    
    # 生成可视化图表
    print("\n📊 生成混淆矩阵...")
    plot_confusion_matrix(final_result['all_labels'], final_result['all_preds'], classes)
    
    print("📈 生成ROC曲线...")
    macro_auc = plot_roc_curve(final_result['all_labels'], final_result['all_outputs'], classes)
    
    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        final_result['all_labels'], final_result['all_preds'], average='weighted'
    )
    
    final_accuracy = 100 * np.sum(np.array(final_result['all_preds']) == np.array(final_result['all_labels'])) / len(final_result['all_labels'])
    
    print(f"\n📊 最终结果:")
    print(f"🎯 测试准确率: {final_accuracy:.2f}%")
    print(f"📊 加权精确率: {precision:.4f}")
    print(f"📊 加权召回率: {recall:.4f}")
    print(f"📊 加权F1分数: {f1:.4f}")
    print(f"📊 宏观平均AUC: {macro_auc:.4f}")
    
    # 清理 Ray
    ray.shutdown()
    
    print("\n✅ 分布式训练完成!")
    print("   - ray_confusion_matrix.png: 混淆矩阵")
    print("   - ray_roc_curve.png: ROC曲线") 
    print("   - ray_best_model.pth: 最佳模型权重")

if __name__ == "__main__":
    main()
