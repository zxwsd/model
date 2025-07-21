import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from model import CNNClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据和标签
data = np.load('data/concatenated_vectors.npy')  # 数据文件路径
# 读取测试集索引和标签

# split_info_path = 'data/dataset_split.xlsx'
# split_info = pd.read_excel(split_info_path, sheet_name='TrainIndicesWithIdAndLabel')
# test_indices = split_info['train_index'].tolist()
# labels = split_info['label'].tolist()

split_info_path = 'data/dataset_split.xlsx'
split_info = pd.read_excel(split_info_path, sheet_name='TestIndicesWithIdAndLabel')
test_indices = split_info['test_index'].tolist()
labels = split_info['label'].tolist()


# split_info_path = 'data/dataset_split.xlsx'
# split_info = pd.read_excel(split_info_path, sheet_name='ValidIndicesWithIdAndLabel')
# test_indices = split_info['valid_index'].tolist()
# labels = split_info['label'].tolist()


# 提取测试数据
data_tensor = torch.tensor(data[test_indices], dtype=torch.long).to(device)  # 提取指定索引的数据并增加通道维度
labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

# 创建数据集和数据加载器
dataset = TensorDataset(data_tensor, labels_tensor)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":
    # 用import 导入时不执行的部分代码
    # 加载最优模型
    input_size = 9924
    num_classes = max(labels) + 1
    model = torch.load('data/model_path/1D/best_model.pth')
    model = model.to(device)
    model.eval()

    # 测试模型
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Sample-wise Predictions and Labels:")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 打印当前 batch 的预测值和真实标签
            for pred, true_label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                print(f"Predicted: {pred}, True Label: {true_label}")

    # 计算并打印整体指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    print("\nOverall Test Results:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))  # 增大图形尺寸
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})  # 增大注释文字字号
    plt.xlabel("Predicted Labels", fontsize=12)  # 增大坐标轴标签字号
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)  # 增大标题字号

    # 保存混淆矩阵图
    save_path = 'data/pic/One'
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=500)  # 增加保存图片的分辨率
    plt.close()