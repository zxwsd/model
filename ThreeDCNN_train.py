import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import os
# 引入3D模型
from model import Molecular3DCNN
# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MolecularDataset3D(Dataset):
    def __init__(self, relative_positions_files, labels, target_size=(56, 56, 3)):
        self.relative_positions_files = relative_positions_files
        self.labels = labels
        self.target_size = target_size
        
    def __len__(self):
        return len(self.relative_positions_files)
    
    def __getitem__(self, idx):
        relative_positions = np.load(self.relative_positions_files[idx])
        current_size = relative_positions.shape[:2]
        target_height, target_width = self.target_size[0], self.target_size[1]

        if current_size[0] < target_height or current_size[1] < target_width:
            pad_height = max(0, target_height - current_size[0])
            pad_width = max(0, target_width - current_size[1])
            relative_positions = np.pad(relative_positions, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        elif current_size[0] > target_height or current_size[1] > target_width:
            relative_positions = relative_positions[:target_height, :target_width, :]
        
        relative_positions = torch.tensor(relative_positions, dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        
        return relative_positions, label



# 加载训练集和验证集索引及标签
split_info_path = 'data/dataset_split.xlsx'
split_info_train = pd.read_excel(split_info_path, sheet_name='TrainIndicesWithIdAndLabel')
split_info_val = pd.read_excel(split_info_path, sheet_name='TestIndicesWithIdAndLabel')

train_indices = split_info_train['train_index'].tolist()
train_labels = split_info_train['label'].tolist()
val_indices = split_info_val['test_index'].tolist()
val_labels = split_info_val['label'].tolist()

# 创建训练集和验证集的文件路径
train_relative_positions_files = [f'data/feature/3D/relative_positions_molecule_{i}.npy' for i in train_indices]
val_relative_positions_files = [f'data/feature/3D/relative_positions_molecule_{i}.npy' for i in val_indices]

# 创建数据集和数据加载器
train_dataset = MolecularDataset3D(train_relative_positions_files, train_labels)
val_dataset = MolecularDataset3D(val_relative_positions_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":
    model = Molecular3DCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    model_save_path = 'data/model_path/3D/best_model.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 定义保存训练过程的文件路径
    training_process_path = 'data/pic/Three/training_process.txt'
    os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

    # 打开文件写入头部信息
    with open(training_process_path, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_Accuracy\n")

    for epoch in range(50):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for relative_positions, labels in train_loader:
            relative_positions, labels = relative_positions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(relative_positions)
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
            for relative_positions, labels in val_loader:
                relative_positions, labels = relative_positions.to(device), labels.to(device)

                outputs = model(relative_positions)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # 将训练过程写入文件
        with open(training_process_path, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.2f}\n")

        # 保存验证集准确率最高的模型，如果验证集准确率相同，则保存训练集准确率最高的模型
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and train_accuracy > best_train_accuracy):
            best_val_accuracy = val_accuracy
            best_train_accuracy = train_accuracy
            torch.save(model, model_save_path)

    print(f"Training complete. Best model saved to {model_save_path} with Val Accuracy: {best_val_accuracy:.4f} and Train Accuracy: {best_train_accuracy:.4f}")