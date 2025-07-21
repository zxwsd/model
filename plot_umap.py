import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Onedata import train_dataset_1d, test_dataset_1d
from Twodata import train_graphs, test_graphs
from Threedata import train_dataset, test_dataset
import matplotlib.pyplot as plt
import numpy as np
import umap  # 引入UMAP库
from model import FusionModel  # 假设模型类定义在 model.py 文件中

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的最优模型
best_model_path = 'data/model_path/1D+2D+3D/best_model.pth'
model = torch.load(best_model_path)
model = model.to(device)
model.eval()

# 加载训练和测试数据
train_loader_1d = TorchDataLoader(train_dataset_1d, batch_size=1, shuffle=False)
train_loader_2d = DataLoader(train_graphs, batch_size=1, shuffle=False)
train_loader_3d = TorchDataLoader(train_dataset, batch_size=1, shuffle=False)

test_loader_1d = TorchDataLoader(test_dataset_1d, batch_size=1, shuffle=False)
test_loader_2d = DataLoader(test_graphs, batch_size=1, shuffle=False)
test_loader_3d = TorchDataLoader(test_dataset, batch_size=1, shuffle=False)

# 合并数据加载器的类
class CombinedLoader:
    def __init__(self, loader_1d, loader_2d, loader_3d):
        self.loader_1d = loader_1d
        self.loader_2d = loader_2d
        self.loader_3d = loader_3d

    def __iter__(self):
        for data_1d, data_2d, data_3d in zip(self.loader_1d, self.loader_2d, self.loader_3d):
            yield data_1d, data_2d, data_3d

# 创建训练集和测试集的数据加载器
train_loader = CombinedLoader(train_loader_1d, train_loader_2d, train_loader_3d)
test_loader = CombinedLoader(test_loader_1d, test_loader_2d, test_loader_3d)

# 初始化特征向量收集和标签收集
all_features = []  # 收集倒数第二层的特征向量
all_labels = []    # 收集标签
is_train = []      # 标记是否为训练集数据

# 定义一个钩子函数来提取特征
def get_features(module, input, output):
    all_features.append(output.cpu().numpy())

# 注册钩子到倒数第二层
# 假设模型的倒数第二层是名为'fc'的全连接层
# 需要根据实际模型结构调整这一行代码
hook = model.fc.register_forward_hook(get_features)

# 提取训练集特征向量
with torch.no_grad():
    for (data_1d, data_2d, data_3d) in train_loader:
        # 1D、2D、3D数据
        x_1d, y_1d = data_1d
        x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d
        x_3d, y_3d = data_3d

        # 2D图数据处理
        edge_index = edge_index[1]
        x_2d_nodes = x_2d_nodes[1]

        # 将数据移动到GPU
        x_1d = x_1d.to(device)
        x_2d_nodes = x_2d_nodes.to(device)
        edge_index = edge_index.to(device)
        x_3d = x_3d.to(device)
        batch_all = batch_2d[1].to(device)
        y_2d_all = y_2d[1].to(device)

        # 模型前向传播
        _ = model(x_1d, x_2d_nodes, edge_index, x_3d, batch_all)

        # 收集标签和训练标记
        all_labels.extend(y_2d_all.cpu().numpy())
        is_train.extend([True] * y_2d_all.size(0))

# 提取测试集特征向量
with torch.no_grad():
    for (data_1d, data_2d, data_3d) in test_loader:
        # 1D、2D、3D数据
        x_1d, y_1d = data_1d
        x_2d_nodes, edge_index, y_2d, batch_2d, ptr = data_2d
        x_3d, y_3d = data_3d

        # 2D图数据处理
        edge_index = edge_index[1]
        x_2d_nodes = x_2d_nodes[1]

        # 将数据移动到GPU
        x_1d = x_1d.to(device)
        x_2d_nodes = x_2d_nodes.to(device)
        edge_index = edge_index.to(device)
        x_3d = x_3d.to(device)
        batch_all = batch_2d[1].to(device)
        y_2d_all = y_2d[1].to(device)

        # 模型前向传播
        _ = model(x_1d, x_2d_nodes, edge_index, x_3d, batch_all)

        # 收集标签和训练标记
        all_labels.extend(y_2d_all.cpu().numpy())
        is_train.extend([False] * y_2d_all.size(0))

# 移除钩子
hook.remove()

# 将特征向量转换为numpy数组
all_features = np.vstack(all_features)

# 使用UMAP进行降维
umap_reducer = umap.UMAP(n_neighbors=400, min_dist=1, random_state=42)
umap_embeddings = umap_reducer.fit_transform(all_features)

# 绘制UMAP可视化图
plt.figure(figsize=(12, 10))

# 绘制训练集数据
train_mask = np.array(is_train)
train_embeddings = umap_embeddings[train_mask]
train_labels = np.array(all_labels)[train_mask]
train_scatter = plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], c=train_labels, cmap='viridis', edgecolor='k', alpha=0.6, label='Train')

# 绘制测试集数据
test_mask = ~np.array(is_train)
test_embeddings = umap_embeddings[test_mask]
test_labels = np.array(all_labels)[test_mask]
test_scatter = plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c=test_labels, cmap='viridis', edgecolor='k', marker='X', label='Test')

# 为测试集的每个点添加数字标签
for i, (x, y) in enumerate(test_embeddings):
    plt.text(x, y, str(i + 1), fontsize=12, weight='bold', color='black', ha='right')

# 创建类别图例
unique_categories = np.unique(train_labels)
category_labels = [f"Level {int(cat)}" for cat in unique_categories]
category_proxies = [plt.Line2D([0], [0], marker='o', color='none', label=label,
                               markerfacecolor=plt.cm.viridis(cat / max(unique_categories)),
                               markeredgecolor='k', markersize=8)
                    for cat, label in zip(unique_categories, category_labels)]

# 创建训练集和测试集图例
train_proxy = plt.Line2D([0], [0], marker='o', color='none', label='Train', markerfacecolor='none', markeredgecolor='k')
test_proxy = plt.Line2D([0], [0], marker='X', color='none', label='Test', markerfacecolor='none', markeredgecolor='k')

# 结合所有图例
all_proxies = [train_proxy, test_proxy] + category_proxies

# 添加图例
plt.legend(handles=all_proxies)

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig('data/pic/kongjianfenbu.png', dpi=500, bbox_inches='tight')
plt.show()