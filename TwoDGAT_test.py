import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import TwoDGAT


# 加载数据和标签
split_info_path = 'data/dataset_split.xlsx'
split_info = pd.read_excel(split_info_path, sheet_name=None)


# 验证集

# val_indices = split_info['TrainIndicesWithIdAndLabel']['train_index'].tolist()
# val_labels = split_info['TrainIndicesWithIdAndLabel']['label'].tolist()

val_indices = split_info['TestIndicesWithIdAndLabel']['test_index'].tolist()
val_labels = split_info['TestIndicesWithIdAndLabel']['label'].tolist()

# val_indices = split_info['ValidIndicesWithIdAndLabel']['valid_index'].tolist()
# val_labels = split_info['ValidIndicesWithIdAndLabel']['label'].tolist()

def load_graph_data(indices, labels):
    graph_data_list = []
    for i, index in enumerate(indices):
        # 加载节点特征和邻接矩阵
        atom_features = np.load(f'data/feature/2D/atom_features_molecule_{index}.npy')
        adjacency_matrix = np.load(f'data/feature/2D/adjacency_matrix_molecule_{index}.npy')

        # 转换为 PyTorch 张量
        x = torch.tensor(atom_features, dtype=torch.float)
        adj_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        molecule_label = torch.tensor([labels[i]], dtype=torch.long)

        # 构建 Data 对象
        data = Data(x=x, edge_index=edge_index, y=molecule_label)
        graph_data_list.append(data)

    return graph_data_list

# 加载验证集数据
val_graphs = load_graph_data(val_indices, val_labels)

# 创建 DataLoader
batch_size = 8
data_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    # 用import 导入时不执行的部分代码
    # 加载最优模型
    best_model_path = 'data/model_path/2D/best_model.pth'
    model = torch.load(best_model_path)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # 在测试集上评估模型
    all_preds = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)

            # 前向传播
            output = model(batch_data.x, batch_data.edge_index)  # 节点级输出 [num_nodes, num_classes]

            # 全局池化，将节点级输出聚合为图级输出
            graph_output = global_mean_pool(output, batch_data.batch)  # 图级输出 [num_graphs, num_classes]

            # 获取预测结果
            preds = graph_output.argmax(dim=-1).cpu().numpy()
            labels = batch_data.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # 打印每个样本的预测标签和真实标签
    for i in range(len(all_labels)):
        print(f"Sample {i + 1}, Predicted Label: {all_preds[i]}, True Label: {all_labels[i]}")

    # 将预测结果和真实标签保存到文件
    result_path = 'data/pic/Two'
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, 'predictions.txt')
    with open(result_file, 'w') as f:
        for i in range(len(all_labels)):
            f.write(f"Sample {i + 1}, Predicted Label: {all_preds[i]}, True Label: {all_labels[i]}\n")

    # 计算分类评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    # 打印评估结果
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))  # 增大图形尺寸
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})  # 增大注释文字字号
    plt.xlabel("Predicted Labels", fontsize=12)  # 增大坐标轴标签字号
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)  # 增大标题字号

    # 保存混淆矩阵图
    save_path = 'data/pic/Two'
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=500)  # 增加保存图片的分辨率
    plt.close()