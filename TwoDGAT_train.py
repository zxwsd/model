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
from model import TwoDGAT


# 加载数据和标签
split_info_path = 'data/dataset_split.xlsx'
split_info = pd.read_excel(split_info_path, sheet_name=None)

# 训练集
train_indices = split_info['TrainIndicesWithIdAndLabel']['train_index'].tolist()
train_labels = split_info['TrainIndicesWithIdAndLabel']['label'].tolist()

# 验证集
val_indices = split_info['TestIndicesWithIdAndLabel']['test_index'].tolist()
val_labels = split_info['TestIndicesWithIdAndLabel']['label'].tolist()

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

# 加载训练集和验证集数据
train_graphs = load_graph_data(train_indices, train_labels)
val_graphs = load_graph_data(val_indices, val_labels)

# 创建 DataLoader
batch_size = 8
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型实例
    in_features = train_graphs[0].x.size(1)
    print(in_features)
    model = TwoDGAT(in_features=train_graphs[0].x.size(1), out_features=128, num_classes=4).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 使用 ReduceLROnPlateau 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-9)
    
    criterion = nn.NLLLoss()

    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    best_model_path = 'data/model_path/2D/best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # 定义保存训练过程的文件路径
    training_process_path = 'data/pic/Two/training_process.txt'
    os.makedirs(os.path.dirname(training_process_path), exist_ok=True)

    # 打开文件写入头部信息
    with open(training_process_path, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_Accuracy\n")

    # 训练和验证
    num_epochs = 50
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data.x, batch_data.edge_index)
            graph_outputs = global_mean_pool(outputs, batch_data.batch)

            loss = criterion(graph_outputs, batch_data.y.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(graph_outputs, 1)
            correct += (predicted == batch_data.y.view(-1)).sum().item()
            total += batch_data.y.size(0)
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)

                outputs = model(batch_data.x, batch_data.edge_index)
                graph_outputs = global_mean_pool(outputs, batch_data.batch)

                loss = criterion(graph_outputs, batch_data.y.view(-1))
                val_loss += loss.item()

                _, predicted = torch.max(graph_outputs, 1)
                correct += (predicted == batch_data.y.view(-1)).sum().item()
                total += batch_data.y.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        # 更新学习率调度器
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # 将训练过程写入文件
        with open(training_process_path, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.2f}\n")

        # 保存验证集准确率最高的模型，如果验证集准确率相同，则保存训练集准确率最高的模型
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and train_accuracy > best_train_accuracy):
            best_val_accuracy = val_accuracy
            best_train_accuracy = train_accuracy
            torch.save(model, best_model_path)

    print(f"Training complete. Best model saved at {best_model_path} with Val Accuracy: {best_val_accuracy:.4f} and Train Accuracy: {best_train_accuracy:.4f}")