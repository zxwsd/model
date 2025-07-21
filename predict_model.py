import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from OneDCNN_test import dataset
from TwoDGAT_test import val_graphs
from ThreeDCNN_test import test_dataset

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的最优模型
from model import FusionModel  # 假设模型类定义在 model.py 文件中
best_model_path = 'data/model_path/1D+2D+3D/best_model.pth'
model = torch.load(best_model_path)
model = model.to(device)
model.eval()

# 加载测试数据
test_loader_1d = TorchDataLoader(dataset, batch_size=1, shuffle=False)
test_loader_2d = DataLoader(val_graphs, batch_size=1, shuffle=False)
test_loader_3d = TorchDataLoader(test_dataset, batch_size=1, shuffle=False)

# 合并测试数据加载器
class CombinedTestLoader:
    def __init__(self, loader_1d, loader_2d, loader_3d):
        self.loader_1d = loader_1d
        self.loader_2d = loader_2d
        self.loader_3d = loader_3d

    def __iter__(self):
        for data_1d, data_2d, data_3d in zip(self.loader_1d, self.loader_2d, self.loader_3d):
            yield data_1d, data_2d, data_3d

test_loader = CombinedTestLoader(test_loader_1d, test_loader_2d, test_loader_3d)

# 只处理第十个测试用例
with torch.no_grad():
    count = 0  # 用于计数
    for (data_1d, data_2d, data_3d) in test_loader:
        count += 1
        if count == 10:  # 只处理第十个测试用例
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

            # 模型预测
            outputs = model(x_1d, x_2d_nodes, edge_index, x_3d, batch_all)
            _, predicted = torch.max(outputs, 1)

            # 输出预测标签和真实标签
            
      
            print(f"{predicted.cpu().numpy()[0]}")
            break  # 退出循环，只处理第十个测试用例