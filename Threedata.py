import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import os

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 数据路径
data_root = 'data/'

# 读取训练集、验证集和测试集的索引和标签
split_info_path = os.path.join(data_root, 'dataset_split.xlsx')
split_info_train = pd.read_excel(split_info_path, sheet_name='TrainIndicesWithIdAndLabel')
split_info_val = pd.read_excel(split_info_path, sheet_name='ValidIndicesWithIdAndLabel')
split_info_test = pd.read_excel(split_info_path, sheet_name='TestIndicesWithIdAndLabel')

# 训练集
train_indices = split_info_train['train_index'].tolist()
train_labels = split_info_train['label'].tolist()

# 验证集
val_indices = split_info_val['valid_index'].tolist()
val_labels = split_info_val['label'].tolist()

# 测试集
test_indices = split_info_test['test_index'].tolist()
test_labels = split_info_test['label'].tolist()

# 创建训练集、验证集和测试集的文件路径
train_relative_positions_files = [os.path.join(data_root, f'feature/3D/relative_positions_molecule_{i}.npy') for i in train_indices]
val_relative_positions_files = [os.path.join(data_root, f'feature/3D/relative_positions_molecule_{i}.npy') for i in val_indices]
test_relative_positions_files = [os.path.join(data_root, f'feature/3D/relative_positions_molecule_{i}.npy') for i in test_indices]

# 创建数据集
train_dataset = MolecularDataset3D(train_relative_positions_files, train_labels)
val_dataset = MolecularDataset3D(val_relative_positions_files, val_labels)
test_dataset = MolecularDataset3D(test_relative_positions_files, test_labels)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

