import torch
import torch.nn as nn
import torch.nn.functional as F
from model import FusionModel1D3D
from torch.utils.data import DataLoader as TorchDataLoader
from OneDCNN_train import train_dataset_1d, val_dataset_1d
from ThreeDCNN_train import train_dataset, val_dataset
import torch.optim as optim
import os

# 合并1D和3D加载器
class CombinedDataLoader1D3D:
    def __init__(self, loader_1d, loader_3d):
        self.loader_1d = loader_1d
        self.loader_3d = loader_3d
        self.batch_size = loader_1d.batch_size
        self.dataset_size = len(loader_1d.dataset)  # 假设1D和3D数据集大小相等
        self.num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for data_1d, data_3d in zip(self.loader_1d, self.loader_3d):
            yield data_1d, data_3d

    def __len__(self):
        return self.num_batches

# 1D和3D数据加载器
data_loader_1d_train = TorchDataLoader(train_dataset_1d, batch_size=8, shuffle=True)
data_loader_3d_train = TorchDataLoader(train_dataset, batch_size=8, shuffle=True)
combined_loader_train = CombinedDataLoader1D3D(data_loader_1d_train, data_loader_3d_train)

data_loader_1d_val = TorchDataLoader(val_dataset_1d, batch_size=8, shuffle=False)
data_loader_3d_val = TorchDataLoader(val_dataset, batch_size=8, shuffle=False)
combined_loader_val = CombinedDataLoader1D3D(data_loader_1d_val, data_loader_3d_val)

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = FusionModel1D3D(in_features_1d=9924, in_features_3d=56, num_classes=4).to(device)

# 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 创建保存训练过程的文件路径
training_process_path = 'data/pic/1+3/training_process.txt'
os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

# 打开文件写入头部信息
with open(training_process_path, 'w') as f:
    f.write("Epoch\tTrain_Loss\tTrain_Accuracy\n")

# 保存路径
best_model_path = 'data/model_path/1D+3D/best_model.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# 训练过程
num_epochs = 50
best_val_accuracy = 0.0
best_train_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches_train = len(combined_loader_train)

    for data_1d, data_3d in combined_loader_train:
        optimizer.zero_grad()

        # 数据解包
        x_1d, y_1d = data_1d
        x_3d, y_3d = data_3d

        # 数据移动到GPU
        x_1d = x_1d.to(device)
        x_3d = x_3d.to(device)
        y_1d = y_1d.to(device)

        # 前向传播
        outputs = model(x_1d, x_3d)
        loss = criterion(outputs, y_1d)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_1d.size(0)
        correct += (predicted == y_1d).sum().item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / num_batches_train

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    num_batches_val = len(combined_loader_val)

    with torch.no_grad():
        for data_1d, data_3d in combined_loader_val:
            # 数据解包
            x_1d, y_1d = data_1d
            x_3d, y_3d = data_3d

            # 数据移动到GPU
            x_1d = x_1d.to(device)
            x_3d = x_3d.to(device)
            y_1d = y_1d.to(device)

            # 前向传播
            outputs = model(x_1d, x_3d)
            loss = criterion(outputs, y_1d)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_1d.size(0)
            val_correct += (predicted == y_1d).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= num_batches_val

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # 将训练过程写入文件
    with open(training_process_path, 'a') as f:
        f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.2f}\n")

    # 更新学习率调度器
    scheduler.step(val_loss)

    # 保存验证集准确率最高的模型，如果验证集准确率相同，则保存训练集准确率最高的模型
    if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and train_accuracy > best_train_accuracy):
        best_val_accuracy = val_accuracy
        best_train_accuracy = train_accuracy
        torch.save(model, best_model_path)

print(f"Training complete. Best model saved with Val Accuracy: {best_val_accuracy:.4f}% and Train Accuracy: {best_train_accuracy:.4f}%")