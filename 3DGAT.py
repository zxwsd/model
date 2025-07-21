import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # 线性变换
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, h, adj):
        # h: 输入特征矩阵, adj: 邻接矩阵
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._attention(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
    
    def _attention(self, Wh):
        a_input = self._prepare_attention_input(Wh)
        e = torch.matmul(a_input, self.a).squeeze(2)
        return self.leakyrelu(e)
    
    def _prepare_attention_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_interleaved = Wh.repeat(N, 1)
        all_pairs = torch.cat([Wh_repeated, Wh_repeated_interleaved], dim=1)
        return all_pairs.view(N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GATLayer(nfeat, nhid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)
class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super(SpatialAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # 线性变换
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features + 3, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, h, pos):
        # h: 输入特征矩阵 (N, in_features)
        # pos: 三维相对位置矩阵 (N, N, 3)
        Wh = torch.mm(h, self.W)  # Wh.shape: (N, out_features)
        print("h.shape:", h.shape)
        print("self.W.shape:", self.W.shape)
        print("Wh.shape:", Wh.shape)
        e = self._attention(Wh, pos)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
    
    def _attention(self, Wh, pos):
        N = Wh.size()[0]
        pos_repeated = pos.repeat(1, N, 1).view(N, N, 3)
        pos_repeated_interleaved = pos.repeat(N, 1, 1).view(N, N, 3)
        pos_diff = pos_repeated - pos_repeated_interleaved
        pos_diff = pos_diff.view(N, N, -1)
        
        a_input = torch.cat([Wh.repeat(N, 1), Wh.repeat_interleave(N, dim=0)], dim=1)
        a_input = torch.cat([a_input, pos_diff.view(-1, 3)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features + 3)
        e = torch.matmul(a_input, self.a).squeeze(2)
        return self.leakyrelu(e)

class SpatialGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, nheads=8):
        super(SpatialGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [SpatialAttentionLayer(nfeat, nhid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpatialAttentionLayer(nhid * nheads, nclass, dropout=dropout)
    
    def forward(self, x, pos):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, pos) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, pos)
        return F.log_softmax(x, dim=1)

class HoloMol(nn.Module):
    def __init__(self, nfeat_2d, nfeat_3d, nhid, nclass, dropout=0.6, nheads=8):
        super(HoloMol, self).__init__()
        self.gat_2d = GAT(nfeat_2d, nhid, nclass, dropout, nheads)
        self.gat_3d = SpatialGAT(nfeat_3d, nhid, nclass, dropout, nheads)
        self.fusion = FusionLayer(nhid * 2, nclass)
    
    def forward(self, x_2d, adj_2d, x_3d, pos_3d):
        x_2d = self.gat_2d(x_2d, adj_2d)
        x_3d = self.gat_3d(x_3d, pos_3d)
        x = torch.cat([x_2d, x_3d], dim=1)
        x = self.fusion(x)
        return F.log_softmax(x, dim=1)

class FusionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FusionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
    
    def forward(self, h):
        return torch.mm(h, self.W)

# 测试代码
if __name__ == "__main__":
    # 加载数据
    atom_features = np.load('data/feature/2D/atom_features_molecule_0.npy')
    adjacency_matrix = np.load('data/feature/2D/adjacency_matrix_molecule_0.npy')
    relative_positions = np.load('data/feature/3D/relative_positions_molecule_0.npy')

    # 转换为 PyTorch 张量
    atom_features = torch.tensor(atom_features, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    relative_positions = torch.tensor(relative_positions, dtype=torch.float32)

    # 初始化模型
    nfeat_2d = atom_features.shape[1]  # 2D特征的维度
    nfeat_3d = relative_positions.shape[2]  # 3D特征的维度
    nhid = 64  # 隐藏层维度
    nclass = 4  # 分类类别数量

    model = HoloMol(nfeat_2d=nfeat_2d, nfeat_3d=nfeat_3d, nhid=nhid, nclass=nclass)

    # 前向传播
    output = model(atom_features, adjacency_matrix, atom_features, relative_positions)

    print(output)
    