import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

# 引入1D模型
from model import CNNClassifier

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据和标签
data = np.load('data/concatenated_vectors.npy')  # 数据文件路径
labels = np.load('data/label.npy')  # 标签文件路径

# 读取训练集索引和标签
split_info_path = 'data/dataset_split.xlsx'
split_info = pd.read_excel(split_info_path, sheet_name=None)

# 训练集
train_indices = split_info['TrainIndicesWithIdAndLabel']['train_index'].tolist()
train_labels = split_info['TrainIndicesWithIdAndLabel']['label'].tolist()

# 验证集
val_indices = split_info['TestIndicesWithIdAndLabel']['test_index'].tolist()
val_labels = split_info['TestIndicesWithIdAndLabel']['label'].tolist()

# 转换为张量
train_data_tensor = torch.tensor(data[train_indices], dtype=torch.long).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
val_data_tensor = torch.tensor(data[val_indices], dtype=torch.long).to(device)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(device)

# 创建数据集和数据加载器
train_dataset_1d = TensorDataset(train_data_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset_1d, batch_size=16, shuffle=False)

val_dataset_1d = TensorDataset(val_data_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset_1d, batch_size=16, shuffle=False)

# 主程序
if __name__ == "__main__":
    input_size = 9924  # 稀疏向量的原始长度
    embedding_dim = 150  # 嵌入维度
    num_classes = max(train_labels + val_labels) + 1

    # 创建模型实例
    model = CNNClassifier(num_classes=num_classes, input_size=input_size, embedding_dim=embedding_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    num_epochs = 50
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0  # 用于记录最佳训练准确率
    best_model_path = 'data/model_path/1D/best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # 创建保存训练过程的文件路径
    training_process_path = 'data/pic/One/training_process.txt'
    os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

    # 打开文件，准备写入训练过程
    with open(training_process_path, 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\n")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # 将训练过程写入文件
        with open(training_process_path, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.2f}\n")

        # 保存验证集准确率最高的模型，如果验证集准确率相同，则保存训练集准确率最高的模型
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and train_accuracy > best_train_accuracy):
            best_val_accuracy = val_accuracy
            best_train_accuracy = train_accuracy
            torch.save(model, best_model_path)

    print(f"Training complete. Best model saved at {best_model_path} with Val Accuracy: {best_val_accuracy:.4f} and Train Accuracy: {best_train_accuracy:.4f}")