import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_geometric.nn import GATConv, global_mean_pool



# class FusionModel(nn.Module):
#     def __init__(self, in_features_1d, in_features, out_features, in_features_3d, num_classes=4, num_heads=4, embedding_dim=150):
#         super(FusionModel, self).__init__()

#         # 1D CNN特征提取
#         self.embedding = nn.Embedding(in_features_1d, embedding_dim)
#         self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1_1d = nn.BatchNorm1d(64)
#         self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.dropout = nn.Dropout(p=0.5)

#         # 动态计算全连接层的输入大小
#         example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)
#         with torch.no_grad():
#             example_output_1d = self._forward_1d_features(example_input_1d)
#         self.fc1d_input_size = example_output_1d.view(-1).size(0)

#         # 2D GAT特征提取
#         # 第一层 GAT
#         self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
#         self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

#         # 第二层 GAT
#         self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
#         self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
#         # 第三层 GAT
#         self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
#         self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
#         # 第四层 GAT
#         self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
#         self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
#         # 第五层 GAT
#         self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
#         self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
#         # 第六层 GAT
#         self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
#         self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
#         # 第七层 GAT
#         self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
#         self.bn7 = BatchNorm(out_features)  # 批归一化
#         self.fc1_2d = nn.Linear(out_features, out_features)

#         # 3D CNN特征提取
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.fc1_input_size = self.calculate_fc1_input_size()
#         self.fc1 = nn.Linear(self.fc1_input_size, 128)

#         # 统一目标维度
#         target_dim = 256
#         self.fc1d = nn.Linear(self.fc1d_input_size, target_dim)
#         self.fc2d = nn.Linear(128, target_dim)  # 对 x_2d 升维
#         self.fc3d = nn.Linear(128, target_dim)  # 对 x_3d 降维

#         # Attention机制的参数
#         self.attention_1d = nn.Parameter(torch.ones(1))
#         self.attention_2d = nn.Parameter(torch.ones(1))
#         self.attention_3d = nn.Parameter(torch.ones(1))
        
#         # 分类层
#         self.fc = nn.Linear(target_dim, num_classes)

#     def calculate_fc1_input_size(self):
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, 56, 56, 3)
#             x = self.conv1(dummy_input)
#             x = self.conv2(x)
#             x = self.conv3(x)
#             return x.numel()
#     def _forward_1d_features(self, x):
#         x = self.embedding(x).permute(0, 2, 1)
#         x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
#         x = F.relu(self.conv2_1d(x))
#         return x

#     def forward(self, x_1d, x, edge_index, x_3d, batch):
#         # 1D CNN特征提取
#         x_1d = self._forward_1d_features(x_1d)
#         x_1d = x_1d.view(x_1d.size(0), -1)
#         x_1d = F.relu(self.fc1d(x_1d))
#         x_1d = self.dropout(x_1d)

#         # 2D GAT特征提取
#         # 第一层 GAT
#         x = F.relu(self.bn1(self.gat1(x, edge_index)))
#         # x = self.dropout(x)

#         # 第二层 GAT
#         x = F.relu(self.bn2(self.gat2(x, edge_index)))
#         # x = self.dropout(x)

#         # 第三层 GAT
#         x = F.relu(self.bn3(self.gat3(x, edge_index)))
#         # x = self.dropout(x)
        
#         # 第四层 GAT
#         x = F.relu(self.bn4(self.gat4(x, edge_index)))
#         # x = self.dropout(x)

#         # 第五层 GAT
#         x = F.relu(self.bn5(self.gat5(x, edge_index)))
#         # x = self.dropout(x)
        
#         # 第六层 GAT
#         x = F.relu(self.bn6(self.gat6(x, edge_index)))
#         # x = self.dropout(x)
        
#         # 第七层 GAT
#         x = F.relu(self.bn7(self.gat7(x, edge_index)))
#         # x = self.dropout(x)
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.fc1_2d(x))
#         x = F.relu(self.fc2d(x))

#          # 3D CNN特征提取
#         x_3d = F.relu(self.conv1(x_3d))
#         x_3d = F.relu(self.conv2(x_3d))
#         x_3d = F.relu(self.conv3(x_3d))
#         x_3d = x_3d.view(x_3d.size(0), -1)
#         x_3d = F.relu(self.fc1(x_3d))
#         x_3d = F.relu(self.fc3d(x_3d))
#         # Attention加权融合
#         fusion = self.attention_1d * x_1d+self.attention_2d * x + self.attention_3d * x_3d
       
#         # print(fusion.shape)
#         # 分类层
#         output = self.fc(fusion)
    
#         return output

class FusionModel(nn.Module):
    def __init__(self, in_features_1d, in_features, out_features, in_features_3d, num_classes=4, num_heads=4, embedding_dim=150):
        super(FusionModel, self).__init__()

        # 1D CNN特征提取
        self.embedding = nn.Embedding(in_features_1d, embedding_dim)
        self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1d = nn.BatchNorm1d(64)
        self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # 动态计算全连接层的输入大小
        example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)
        with torch.no_grad():
            example_output_1d = self._forward_1d_features(example_input_1d)
        self.fc1d_input_size = example_output_1d.view(-1).size(0)

        # 2D GAT特征提取
        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化
        self.fc1_2d = nn.Linear(out_features, out_features)

        # 3D CNN特征提取
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        # 统一目标维度
        target_dim = 256
        self.fc1d = nn.Linear(self.fc1d_input_size, target_dim)
        self.fc2d = nn.Linear(128, target_dim)  # 对 x_2d 升维
        self.fc3d = nn.Linear(128, target_dim)  # 对 x_3d 降维

        # Attention机制的参数
        self.attention_1d = nn.Parameter(torch.ones(1))
        self.attention_2d = nn.Parameter(torch.ones(1))
        self.attention_3d = nn.Parameter(torch.ones(1))
        
        # 分类层
        self.fc = nn.Linear(target_dim, num_classes)

       
        
    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()
    
    def _forward_1d_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
        x = F.relu(self.conv2_1d(x))
        return x

    def forward(self, x_1d, x, edge_index, x_3d, batch):
        # 1D CNN特征提取
        x_1d = self._forward_1d_features(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x_1d = F.relu(self.fc1d(x_1d))
        x_1d = self.dropout(x_1d)

        # 2D GAT特征提取
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1_2d(x))
        x = F.relu(self.fc2d(x))

         # 3D CNN特征提取
        x_3d = F.relu(self.conv1(x_3d))
        x_3d = F.relu(self.conv2(x_3d))
        x_3d = F.relu(self.conv3(x_3d))
        x_3d = x_3d.view(x_3d.size(0), -1)
        x_3d = F.relu(self.fc1(x_3d))
        x_3d = F.relu(self.fc3d(x_3d))
        # Attention加权融合
        fusion = self.attention_1d * x_1d + self.attention_2d * x + self.attention_3d * x_3d
       
        # 分类层
        output = self.fc(fusion)
    
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class FusionModelconcat(nn.Module):
    def __init__(self, in_features_1d, in_features, out_features, in_features_3d, num_classes=4, num_heads=4, embedding_dim=150):
        super(FusionModelconcat, self).__init__()

        # 1D CNN特征提取
        self.embedding = nn.Embedding(in_features_1d, embedding_dim)
        self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1d = nn.BatchNorm1d(64)
        self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # 动态计算全连接层的输入大小
        example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)
        with torch.no_grad():
            example_output_1d = self._forward_1d_features(example_input_1d)
        self.fc1d_input_size = example_output_1d.view(-1).size(0)
        self.fc1d = nn.Linear(self.fc1d_input_size, 128)

        # 2D GAT特征提取
        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化
        self.fc1_2d = nn.Linear(out_features, 128)

        # 3D CNN特征提取
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        # 统一目标维度
        self.fc2d = nn.Linear(128, 128)  # 对 x_2d 维度调整

        # 融合层
        self.fc_fusion = nn.Linear(128 * 3, 256)

        # 分类层
        self.fc = nn.Linear(256, num_classes)

    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()
    def _forward_1d_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
        x = F.relu(self.conv2_1d(x))
        return x

    def forward(self, x_1d, x, edge_index, x_3d, batch):
        # 1D CNN特征提取
        x_1d = self._forward_1d_features(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x_1d = F.relu(self.fc1d(x_1d))
        x_1d = self.dropout(x_1d)

        # 2D GAT特征提取
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1_2d(x))
        x = F.relu(self.fc2d(x))

         # 3D CNN特征提取
        x_3d = F.relu(self.conv1(x_3d))
        x_3d = F.relu(self.conv2(x_3d))
        x_3d = F.relu(self.conv3(x_3d))
        x_3d = x_3d.view(x_3d.size(0), -1)
        x_3d = F.relu(self.fc1(x_3d))
        # 将三种特征拼接在一起
        fusion = torch.cat([x_1d, x, x_3d], dim=1)
        fusion = F.relu(self.fc_fusion(fusion))
        # 分类层
        output = self.fc(fusion)
    
        return output   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class FusionModelAdd(nn.Module):
    def __init__(self, in_features_1d, in_features, out_features, in_features_3d, num_classes=4, num_heads=4, embedding_dim=150):
        super(FusionModelAdd, self).__init__()

        # 1D CNN特征提取
        self.embedding = nn.Embedding(in_features_1d, embedding_dim)
        self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1d = nn.BatchNorm1d(64)
        self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # 动态计算全连接层的输入大小
        example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)
        with torch.no_grad():
            example_output_1d = self._forward_1d_features(example_input_1d)
        self.fc1d_input_size = example_output_1d.view(-1).size(0)
        self.fc1d = nn.Linear(self.fc1d_input_size, 128)

        # 2D GAT特征提取
        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化
        self.fc1_2d = nn.Linear(out_features, 128)

        # 3D CNN特征提取
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        # 调整各模态特征到相同的维度以便求和融合
        self.fc2d = nn.Linear(128, 128)  # 对 x_2d 维度调整
        self.fc3d = nn.Linear(128, 128)  # 对 x_3d 维度调整

        # 分类层
        self.fc = nn.Linear(128, num_classes)

    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()
    def _forward_1d_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
        x = F.relu(self.conv2_1d(x))
        return x

    def forward(self, x_1d, x, edge_index, x_3d, batch):
        # 1D CNN特征提取
        x_1d = self._forward_1d_features(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x_1d = F.relu(self.fc1d(x_1d))
        x_1d = self.dropout(x_1d)

        # 2D GAT特征提取
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1_2d(x))
        x = F.relu(self.fc2d(x))

         # 3D CNN特征提取
        x_3d = F.relu(self.conv1(x_3d))
        x_3d = F.relu(self.conv2(x_3d))
        x_3d = F.relu(self.conv3(x_3d))
        x_3d = x_3d.view(x_3d.size(0), -1)
        x_3d = F.relu(self.fc1(x_3d))
        x_3d = F.relu(self.fc3d(x_3d))
        # 特征求和融合
        fusion = x_1d + x + x_3d
        # 分类层
        output = self.fc(fusion)
    
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
class FusionModel1D2D(nn.Module):
    def __init__(self, in_features_1d,in_features, out_features, num_classes=4, num_heads=4, embedding_dim=150):
        super(FusionModel1D2D, self).__init__()

        # 1D CNN特征提取
        self.embedding = nn.Embedding(in_features_1d, embedding_dim)
        self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1d = nn.BatchNorm1d(64)
        self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # 动态计算全连接层的输入大小
        example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)
        with torch.no_grad():
            example_output_1d = self._forward_1d_features(example_input_1d)
        self.fc1d_input_size = example_output_1d.view(-1).size(0)

        # 2D GAT特征提取
        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化
        self.fc1 = nn.Linear(out_features, out_features)
        # 特征融合
        target_dim = 256
        self.fc1d = nn.Linear(self.fc1d_input_size, target_dim)
        self.fc2d = nn.Linear(128, target_dim)

        # Attention机制的参数
        self.attention_1d = nn.Parameter(torch.ones(1))
        self.attention_2d = nn.Parameter(torch.ones(1))

        # 分类层
        self.fc = nn.Linear(target_dim, num_classes)
       
    def _forward_1d_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
        x = F.relu(self.conv2_1d(x))
        return x

    def forward(self, x_1d, x, edge_index,batch):
        # 1D CNN特征提取
        x_1d = self._forward_1d_features(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x_1d = F.relu(self.fc1d(x_1d))
        x_1d = self.dropout(x_1d)

        # 2D GAT特征提取
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2d(x))
        
        # Attention加权融合
        
        fusion = self.attention_1d * x_1d + self.attention_2d * x

        # 分类层
        output = self.fc(fusion)
        return output

class FusionModel2D3D(nn.Module):
    def __init__(self, in_features, out_features,in_features_3d, num_classes=4, num_heads=8):
        super(FusionModel2D3D, self).__init__()

        # 2D GAT特征提取
        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化
        self.fc1_2d = nn.Linear(out_features, out_features)

        # 3D CNN特征提取
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        

        # 统一目标维度
        target_dim = 256
        self.fc2d = nn.Linear(128, target_dim)  # 对 x_2d 升维
        self.fc3d = nn.Linear(128, target_dim)  # 对 x_3d 降维

        # Attention机制的参数
        self.attention_2d = nn.Parameter(torch.ones(1))
        self.attention_3d = nn.Parameter(torch.ones(1))
        
        # 分类层
        self.fc = nn.Linear(target_dim, num_classes)

    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()
    def forward(self, x, edge_index, x_3d, batch):
        # 2D GAT特征提取
        # 2D GAT特征提取
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1_2d(x))
        x = F.relu(self.fc2d(x))
        
        # 3D CNN特征提取
        x_3d = F.relu(self.conv1(x_3d))
        x_3d = F.relu(self.conv2(x_3d))
        x_3d = F.relu(self.conv3(x_3d))
        x_3d = x_3d.view(x_3d.size(0), -1)
        x_3d = F.relu(self.fc1(x_3d))
        x_3d = F.relu(self.fc3d(x_3d))
        # Attention加权融合
        fusion = self.attention_2d * x + self.attention_3d * x_3d
        
        # 分类层
        output = self.fc(fusion)
        return output
    
class FusionModel1D3D(nn.Module):
    def __init__(self, in_features_1d, in_features_3d, num_classes=4, embedding_dim=150):
        super(FusionModel1D3D, self).__init__()

        # 1D CNN特征提取
        self.embedding = nn.Embedding(in_features_1d, embedding_dim)
        self.conv1_1d = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1d = nn.BatchNorm1d(64)
        self.pool_1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1d = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # 动态计算1D特征的全连接层输入大小
        example_input_1d = torch.zeros(1, in_features_1d, dtype=torch.long)  # 示例输入
        with torch.no_grad():
            example_output_1d = self._forward_1d_features(example_input_1d)
        self.fc1d_input_size = example_output_1d.view(-1).size(0)

        # 3D CNN特征提取
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        target_dim = 256  # 统一目标维度
        self.fc1d = nn.Linear(self.fc1d_input_size, target_dim)  # 对 x_1d 降维
        self.fc3d = nn.Linear(128, target_dim)  # 对 x_3d 降维

        # Attention机制的参数
        self.attention_1d = nn.Parameter(torch.ones(1))
        self.attention_3d = nn.Parameter(torch.ones(1))

        # 分类层
        self.fc = nn.Linear(target_dim, num_classes)

    def _forward_1d_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # 嵌入层的输出调整为 (batch, channels, seq_len)
        x = self.pool_1d(F.relu(self.bn1_1d(self.conv1_1d(x))))
        x = F.relu(self.conv2_1d(x))
        return x
    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()
    def forward(self, x_1d, x_3d):
        # 1D CNN特征提取
        x_1d = self._forward_1d_features(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)  # 展平
        x_1d = F.relu(self.fc1d(x_1d))
        x_1d = self.dropout(x_1d)

        # 3D CNN特征提取
        x_3d = F.relu(self.conv1(x_3d))
        x_3d = F.relu(self.conv2(x_3d))
        x_3d = F.relu(self.conv3(x_3d))
        x_3d = x_3d.view(x_3d.size(0), -1)
        x_3d = F.relu(self.fc1(x_3d))
        x_3d = F.relu(self.fc3d(x_3d))

        # Attention加权融合
        fusion = self.attention_1d * x_1d + self.attention_3d * x_3d

        # 分类层
        output = self.fc(fusion)
        return output
    
# 定义 1D CNN 模型
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, input_size, embedding_dim):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        # 动态计算全连接层的输入大小
        example_input = torch.zeros(1, input_size, dtype=torch.long)  # 示例输入
        with torch.no_grad():
            example_output = self._forward_features(example_input)
        fc_input_size = example_output.view(-1).size(0)

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def _forward_features(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # 嵌入层的输出调整为 (batch, channels, seq_len)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
class TwoDGAT(nn.Module):
    def __init__(self, in_features, out_features, num_classes=4, num_heads=4, dropout=0.01):
        super(TwoDGAT, self).__init__()

        # 第一层 GAT
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.bn1 = BatchNorm(out_features * num_heads)  # 批归一化

        # 第二层 GAT
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn2 = BatchNorm(out_features * num_heads)  # 批归一化
      
        # 第三层 GAT
        self.gat3 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn3 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第四层 GAT
        self.gat4 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn4 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第五层 GAT
        self.gat5 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn5 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第六层 GAT
        self.gat6 = GATConv(out_features * num_heads, out_features, heads=num_heads, concat=True)
        self.bn6 = BatchNorm(out_features * num_heads)  # 批归一化
        
        # 第七层 GAT
        self.gat7 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        self.bn7 = BatchNorm(out_features)  # 批归一化

        # 全局池化层（用于从图到全局表示的映射）
        self.global_pool = global_mean_pool

        # 全连接层
        self.fc1 = nn.Linear(out_features, out_features)
        self.fc2 = nn.Linear(out_features, num_classes)

        # Dropout 和激活函数
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # 第一层 GAT
        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        # x = self.dropout(x)

        # 第二层 GAT
        x = F.relu(self.bn2(self.gat2(x, edge_index)))
        # x = self.dropout(x)

        # 第三层 GAT
        x = F.relu(self.bn3(self.gat3(x, edge_index)))
        # x = self.dropout(x)
        
        # 第四层 GAT
        x = F.relu(self.bn4(self.gat4(x, edge_index)))
        # x = self.dropout(x)

        # 第五层 GAT
        x = F.relu(self.bn5(self.gat5(x, edge_index)))
        # x = self.dropout(x)
        
        # 第六层 GAT
        x = F.relu(self.bn6(self.gat6(x, edge_index)))
        # x = self.dropout(x)
        
        # 第七层 GAT
        x = F.relu(self.bn7(self.gat7(x, edge_index)))
        # x = self.dropout(x)


        # # 全局池化
        # x = self.global_pool(x, batch)

        # 全连接层
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
    
class Molecular3DCNN(nn.Module):
    def __init__(self):
        super(Molecular3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 4)
        
    def calculate_fc1_input_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 56, 56, 3)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()

    def forward(self, relative_positions):
        x = F.relu(self.conv1(relative_positions))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
