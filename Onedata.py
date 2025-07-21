import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
data_root = 'data/'

# 加载数据和标签
data = np.load(os.path.join(data_root, 'concatenated_vectors.npy'))  # 数据文件路径
labels = np.load(os.path.join(data_root, 'label.npy'))  # 标签文件路径

# 读取训练集、验证集和测试集的索引和标签
split_info_path = os.path.join(data_root, 'dataset_split.xlsx')
split_info = pd.read_excel(split_info_path, sheet_name=None)

# 训练集
train_indices = split_info['TrainIndicesWithIdAndLabel']['train_index'].tolist()
train_labels = split_info['TrainIndicesWithIdAndLabel']['label'].tolist()

# 验证集
val_indices = split_info['ValidIndicesWithIdAndLabel']['valid_index'].tolist()
val_labels = split_info['ValidIndicesWithIdAndLabel']['label'].tolist()

# 测试集
test_indices = split_info['TestIndicesWithIdAndLabel']['test_index'].tolist()
test_labels = split_info['TestIndicesWithIdAndLabel']['label'].tolist()

# 转换为张量
train_data_tensor = torch.tensor(data[train_indices], dtype=torch.long)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

val_data_tensor = torch.tensor(data[val_indices], dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

test_data_tensor = torch.tensor(data[test_indices], dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# 创建数据集
train_dataset_1d = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset_1d = TensorDataset(val_data_tensor, val_labels_tensor)
test_dataset_1d = TensorDataset(test_data_tensor, test_labels_tensor)

# 创建数据加载器
train_loader = DataLoader(train_dataset_1d, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset_1d, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset_1d, batch_size=8, shuffle=False)

