import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from model import Molecular3DCNN
from torch_geometric.loader import DataLoader as GeometricDataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# 加载测试集索引和标签
# split_info_path = 'data/dataset_split.xlsx'
# split_info = pd.read_excel(split_info_path, sheet_name='TrainIndicesWithIdAndLabel')
# test_indices = split_info['train_index'].tolist()
# test_labels = split_info['label'].tolist()

split_info_path = 'data/dataset_split.xlsx'
split_info = pd.read_excel(split_info_path, sheet_name='TestIndicesWithIdAndLabel')
test_indices = split_info['test_index'].tolist()
test_labels = split_info['label'].tolist()

# split_info_path = 'data/dataset_split.xlsx'
# split_info = pd.read_excel(split_info_path, sheet_name='ValidIndicesWithIdAndLabel')
# test_indices = split_info['valid_index'].tolist()
# test_labels = split_info['label'].tolist()




relative_positions_files = [f'data/feature/3D/relative_positions_molecule_{index}.npy' for index in test_indices]

# 创建测试集数据集和数据加载器
test_dataset = MolecularDataset3D(relative_positions_files, test_labels)
test_loader = GeometricDataLoader(test_dataset, batch_size=8, shuffle=False)

if __name__ == "__main__":
    # 加载最优模型
    model_path = 'data/model_path/3D/best_model.pth'
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 在测试集上评估模型
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for relative_positions, labels in test_loader:
            relative_positions, labels = relative_positions.to(device), labels.to(device)
            outputs = model(relative_positions)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 输出每个样本的预测标签和正确标签
    for i in range(len(all_labels)):
        print(f"Sample {i + 1}, Predicted Label: {all_preds[i]}, True Label: {all_labels[i]}")

    # 将每个样本的预测标签和正确标签保存到文件
    result_path = 'data/pic/Three'
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, 'sample_predictions.txt')
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)

    # 保存混淆矩阵图
    save_path = 'data/pic/Three'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=500)
    plt.close()