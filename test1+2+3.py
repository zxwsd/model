import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Onedata import val_dataset_1d
from Twodata import val_graphs
from Threedata import val_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 使用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载保存的最优模型
from model import FusionModel  # 假设模型类定义在 model.py 文件中
best_model_path = 'data/model_path/1D+2D+3D/best_model.pth'
model = torch.load(best_model_path)
model = model.to(device)
model.eval()

# 加载测试数据
test_loader_1d = TorchDataLoader(val_dataset_1d, batch_size=1, shuffle=False)
test_loader_2d = DataLoader(val_graphs, batch_size=1, shuffle=False)
test_loader_3d = TorchDataLoader(val_dataset, batch_size=1, shuffle=False)

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

# 初始化评估指标
all_labels = []
all_predictions = []

# 测试模型
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

        # 模型预测
        outputs = model(x_1d, x_2d_nodes, edge_index, x_3d, batch_all)
        _, predicted = torch.max(outputs, 1)

        # 收集真实标签和预测标签
        all_labels.extend(y_2d_all.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 计算分类评估指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions)

# 打印评估结果
print("Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# 打印每条数据的预测标签和真实标签
print("\nPer-sample Predictions and Labels:")
for i, (label, prediction) in enumerate(zip(all_labels, all_predictions)):
    print(f"Sample {i+1}: True Label = {label}, Predicted Label = {prediction}")

# 绘制混淆矩阵并保存
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))  # 增大图形尺寸
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})  # 增大注释文字字号
    plt.xlabel("Predicted Labels", fontsize=12)  # 增大坐标轴标签字号
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)  # 增大标题字号
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)  # 增加保存图片的分辨率
    plt.close()

# 确保保存路径存在
save_dir = 'data/pic/1+2+3'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'confusion_matrix.png')

# 绘制并保存混淆矩阵
plot_confusion_matrix(conf_matrix, save_path)
print(f"Confusion matrix saved to {save_path}")