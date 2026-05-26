"""
火龙果茎部病害分类 - ResNet50迁移学习
数据集: Dragon Fruit Stem Disease Dataset
类别: Anthracnose, Brown_Stem_Spot, Gray_Blight, Soft_Rot, Stem_Canker, Healthy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 配置参数 ====================
# 请修改为你的数据集路径
DATA_DIR = r'E:\\VSCode\\火龙果识别项目\\数据\\Dragon Fruit (Pitahaya)\\Original Images'  

NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
IMAGE_SIZE = 224

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if device.type == 'cuda':
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print('='*60)

# ==================== 数据预处理 ====================
print('\n[1/6] 正在加载数据...')

# 训练数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 验证数据（仅标准化）
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 加载数据集
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
class_names = full_dataset.classes

print(f'找到 {len(full_dataset)} 张图像')
print(f'类别: {class_names}')
print(f'各类别样本数:')
for i, class_name in enumerate(class_names):
    class_count = len([img for img, label in full_dataset.samples if label == i])
    print(f'  {class_name}: {class_count} 张')

# 划分训练集和验证集
train_size = int(TRAIN_RATIO * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 为验证集应用不同的变换
val_dataset.dataset.transform = val_transform

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'\n训练集: {len(train_dataset)} 张')
print(f'验证集: {len(val_dataset)} 张')
print('='*60)

# ==================== 构建模型 ====================
print('\n[2/6] 正在构建ResNet50模型...')

def create_model(num_classes):
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换分类头
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

model = create_model(NUM_CLASSES).to(device)

# 统计可训练参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'总参数: {total_params:,}')
print(f'可训练参数: {trainable_params:,}')
print(f'冻结参数: {total_params - trainable_params:,}')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
print('='*60)

# ==================== 训练函数 ====================
print('\n[3/6] 开始训练...\n')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, 'best_model.pth')
        
        print(f'Epoch [{epoch+1:2d}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print(f'\n最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})')
    print(f'最佳模型已保存为: best_model.pth')
    return train_losses, val_losses, train_accs, val_accs

# 执行训练
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
)
print('='*60)

# ==================== 可视化训练结果 ====================
print('\n[4/6] 正在生成训练曲线...')

def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练与验证损失曲线', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(val_accs, 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('训练与验证准确率曲线', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('训练曲线已保存为: training_curves.png')

plot_training_results(train_losses, val_losses, train_accs, val_accs)

# ==================== 模型评估 ====================
print('\n[5/6] 正在评估模型...')

# 加载最佳模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
best_val_acc = checkpoint['val_acc']
print(f'加载最佳模型 (验证准确率: {best_val_acc:.2f}%)')

def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'\n验证集准确率: {accuracy:.2f}%')
    
    # 分类报告
    print('\n分类报告:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 12})
    plt.title('混淆矩阵', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('混淆矩阵已保存为: confusion_matrix.png')
    
    return accuracy, all_preds, all_labels

# 执行评估
final_accuracy, predictions, labels = evaluate_model(model, val_loader, class_names)
print('='*60)

# ==================== 保存最终结果 ====================
print('\n[6/6] 正在保存结果...')

# 保存训练历史
# history = {
#     'train_losses': train_losses,
#     'val_losses': val_losses,
#     'train_accs': train_accs,
#     'val_accs': val_accs,
#     'best_val_acc': best_val_acc,
#     'final_val_acc': final_accuracy,
#     'class_names': class_names,
#     'num_epochs': NUM_EPOCHS,
#     'batch_size': BATCH_SIZE,
#     'learning_rate': LEARNING_RATE
# }

# torch.save(history, 'training_history.pth')

# 打印最终总结
print('\n' + '='*60)
print('训练完成！')
print('='*60)
print(f'最佳验证准确率: {best_val_acc:.2f}%')
print(f'最终验证准确率: {final_accuracy:.2f}%')
print(f'\n保存的文件:')
print(f'  - best_model.pth (最佳模型权重)')
#print(f'  - training_history.pth (训练历史)')
print(f'  - training_curves.png (训练曲线图)')
print(f'  - confusion_matrix.png (混淆矩阵)')
print('='*60)