import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from OneDCNN_train import train_dataset_1d, val_dataset_1d
from TwoDGAT_train import train_graphs, val_graphs
from model import FusionModel1D2D
import os

# 数据加载器
class CombinedDataLoader1D2D:
    def __init__(self, loader_1d, loader_2d):
        self.loader_1d = loader_1d
        self.loader_2d = loader_2d
        self.batch_size = loader_1d.batch_size
        self.dataset_size = len(loader_1d.dataset)
        self.num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for data_1d, data_2d in zip(self.loader_1d, self.loader_2d):
            yield data_1d, data_2d

    def __len__(self):
        return self.num_batches

# 1D、2D数据加载器
data_loader_1d_train = TorchDataLoader(train_dataset_1d, batch_size=8, shuffle=False)
data_loader_2d_train = DataLoader(train_graphs, batch_size=8, shuffle=False)
combined_loader_train = CombinedDataLoader1D2D(data_loader_1d_train, data_loader_2d_train)

data_loader_1d_val = TorchDataLoader(val_dataset_1d, batch_size=8, shuffle=False)
data_loader_2d_val = DataLoader(val_graphs, batch_size=8, shuffle=False)
combined_loader_val = CombinedDataLoader1D2D(data_loader_1d_val, data_loader_2d_val)

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例
model = FusionModel1D2D(in_features_1d=9924, in_features=60, out_features=128, num_classes=4).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

# 保存路径
best_model_path = 'data/model_path/1D+2D/best_model.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# 创建保存训练过程的文件路径
training_process_path = 'data/pic/1+2/training_process.txt'
os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

# 打开文件写入头部信息
with open(training_process_path, 'w') as f:
    f.write("Epoch\tTrain_Loss\tTrain_Accuracy\n")

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

    for (data_1d, data_2d) in combined_loader_train:
        optimizer.zero_grad()

        # 1D、2D数据
        x_1d, y_1d = data_1d
        x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d

        edge_index = edge_index[1]
        x_2d_nodes = x_2d_nodes[1]

        # 数据移动到GPU
        x_1d = x_1d.to(device)
        x_2d_nodes = x_2d_nodes.to(device)
        edge_index = edge_index.to(device)
        batch_all = batch_2d[1].to(device)
        y_2d_all = y_2d[1].to(device)

        # 前向传播
        outputs = model(x_1d, x_2d_nodes, edge_index, batch_all)
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
        for (data_1d, data_2d) in combined_loader_val:
            x_1d, y_1d = data_1d
            x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d

            edge_index = edge_index[1]
            x_2d_nodes = x_2d_nodes[1]

            x_1d = x_1d.to(device)
            x_2d_nodes = x_2d_nodes.to(device)
            edge_index = edge_index.to(device)
            batch_all = batch_2d[1].to(device)
            y_2d_all = y_2d[1].to(device)

            outputs = model(x_1d, x_2d_nodes, edge_index, batch_all)
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
        # 保存整个模型，包括架构和参数
        torch.save(model, best_model_path)

print(f"Training complete. Best model saved with Val Accuracy: {best_val_accuracy:.4f}% and Train Accuracy: {best_train_accuracy:.4f}%")