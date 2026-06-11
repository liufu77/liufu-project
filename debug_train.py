"""
训练调试脚本 - 带详细日志
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import time

# 配置
DATA_DIR = r'd:\项目\实验室-智耕兴农\项目材料\植物精准补光与光温水肥智能协同决策大模型\Original Images'
NUM_CLASSES = 6
BATCH_SIZE = 8
NUM_EPOCHS = 3
IMAGE_SIZE = 224

device = torch.device('cpu')
print(f'使用设备: {device}')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('\n[1] 加载数据集...')
start_time = time.time()
try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f'数据集加载成功，共 {len(dataset)} 张图像')
    print(f'类别: {dataset.classes}')
except Exception as e:
    print(f'数据集加载失败: {e}')
    exit(1)
print(f'耗时: {time.time()-start_time:.2f}秒')

print('\n[2] 划分数据集...')
start_time = time.time()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')
print(f'耗时: {time.time()-start_time:.2f}秒')

print('\n[3] 创建数据加载器...')
start_time = time.time()
try:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'数据加载器创建成功')
    print(f'训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}')
except Exception as e:
    print(f'数据加载器创建失败: {e}')
    exit(1)
print(f'耗时: {time.time()-start_time:.2f}秒')

print('\n[4] 测试数据加载...')
start_time = time.time()
try:
    for i, (images, labels) in enumerate(train_loader):
        print(f'批次 {i+1}: 图片形状 {images.shape}, 标签 {labels}')
        if i >= 2:
            break
    print('数据加载测试通过')
except Exception as e:
    print(f'数据加载测试失败: {e}')
    exit(1)
print(f'耗时: {time.time()-start_time:.2f}秒')

print('\n[5] 构建模型...')
start_time = time.time()
try:
    model = models.mobilenet_v2(weights=None, num_classes=NUM_CLASSES)
    model = model.to(device)
    print('模型构建成功')
except Exception as e:
    print(f'模型构建失败: {e}')
    exit(1)
print(f'耗时: {time.time()-start_time:.2f}秒')

print('\n[6] 训练...')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            print(f'  Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, 耗时: {time.time()-start_time:.2f}秒')

print('\n[7] 保存模型...')
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': dataset.classes
}, 'debug_model.pth')
print('模型保存成功: debug_model.pth')
print('\n训练完成!')
