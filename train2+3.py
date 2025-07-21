import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from model import FusionModel2D3D
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from TwoDGAT_train import train_graphs, val_graphs
from ThreeDCNN_train import train_dataset, val_dataset
import torch.optim as optim
import os

# 合并2D和3D加载器
class CombinedDataLoader2D3D:
    def __init__(self, loader_2d, loader_3d):
        self.loader_2d = loader_2d
        self.loader_3d = loader_3d
        self.batch_size = loader_2d.batch_size
        self.dataset_size = len(loader_2d.dataset)  # 假设2D和3D数据集大小相等
        self.num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for data_2d, data_3d in zip(self.loader_2d, self.loader_3d):
            yield data_2d, data_3d

    def __len__(self):
        return self.num_batches

# 2D和3D数据加载器
data_loader_2d_train = DataLoader(train_graphs, batch_size=8, shuffle=False)
data_loader_3d_train = TorchDataLoader(train_dataset, batch_size=8, shuffle=False)
combined_loader_train = CombinedDataLoader2D3D(data_loader_2d_train, data_loader_3d_train)

data_loader_2d_val = DataLoader(val_graphs, batch_size=8, shuffle=False)
data_loader_3d_val = TorchDataLoader(val_dataset, batch_size=8, shuffle=False)
combined_loader_val = CombinedDataLoader2D3D(data_loader_2d_val, data_loader_3d_val)

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = FusionModel2D3D(in_features=60, out_features=128, in_features_3d=56, num_classes=4).to(device)

# 优化器与损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 创建保存训练过程的文件路径
training_process_path = 'data/pic/2+3/training_process.txt'
os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

# 打开文件写入头部信息
with open(training_process_path, 'w') as f:
    f.write("Epoch\tTrain_Loss\tTrain_Accuracy\n")

# 保存路径
best_model_path = 'data/model_path/2D+3D/best_model.pth'
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

    for data_2d, data_3d in combined_loader_train:
        optimizer.zero_grad()

        # 2D数据
        x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d
        # 3D数据
        x_3d, y_3d = data_3d

        edge_index = edge_index[1]
        x_2d_nodes = x_2d_nodes[1]

        # 数据移动到GPU
        x_2d_nodes = x_2d_nodes.to(device)
        edge_index = edge_index.to(device)
        batch_all = batch_2d[1].to(device)
        x_3d = x_3d.to(device)
        y_2d_all = y_2d[1].to(device)

        # 前向传播
        outputs = model(x_2d_nodes, edge_index, x_3d, batch_all)
        loss = criterion(outputs, y_2d_all)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_2d_all.size(0)
        correct += (predicted == y_2d_all).sum().item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / num_batches_train

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    num_batches_val = len(combined_loader_val)

    with torch.no_grad():
        for data_2d, data_3d in combined_loader_val:
            # 2D数据
            x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d
            # 3D数据
            x_3d, y_3d = data_3d

            edge_index = edge_index[1]
            x_2d_nodes = x_2d_nodes[1]

            # 数据移动到GPU
            x_2d_nodes = x_2d_nodes.to(device)
            edge_index = edge_index.to(device)
            batch_all = batch_2d[1].to(device)
            x_3d = x_3d.to(device)
            y_2d_all = y_2d[1].to(device)

            # 前向传播
            outputs = model(x_2d_nodes, edge_index, x_3d, batch_all)
            loss = criterion(outputs, y_2d_all)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_2d_all.size(0)
            val_correct += (predicted == y_2d_all).sum().item()

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