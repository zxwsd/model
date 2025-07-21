import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
data_root = 'data/'

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

def load_graph_data(indices, labels):
    graph_data_list = []
    for i, index in enumerate(indices):
        # 加载节点特征和邻接矩阵
        atom_features = np.load(os.path.join(data_root, f'feature/2D/atom_features_molecule_{index}.npy'))
        adjacency_matrix = np.load(os.path.join(data_root, f'feature/2D/adjacency_matrix_molecule_{index}.npy'))

        # 转换为 PyTorch 张量
        x = torch.tensor(atom_features, dtype=torch.float)
        adj_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        molecule_label = torch.tensor([labels[i]], dtype=torch.long)

        # 构建 Data 对象
        data = Data(x=x, edge_index=edge_index, y=molecule_label)
        graph_data_list.append(data)

    return graph_data_list

# 加载训练集、验证集和测试集数据
train_graphs = load_graph_data(train_indices, train_labels)
val_graphs = load_graph_data(val_indices, val_labels)
test_graphs = load_graph_data(test_indices, test_labels)

# 创建数据加载器
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=False)
val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

