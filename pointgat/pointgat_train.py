import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PointGAT_Layers import PointGAT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 假设你的数据集已经准备好，这里是一个示例数据集类
class MolecularDataset(Dataset):
    def __init__(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature, labels):
        self.atom_list = atom_list
        self.bond_list = bond_list
        self.atom_degree_list = atom_degree_list
        self.bond_degree_list = bond_degree_list
        self.atom_mask = atom_mask
        self.xyz_feature = xyz_feature
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.atom_list[idx],
            self.bond_list[idx],
            self.atom_degree_list[idx],
            self.bond_degree_list[idx],
            self.atom_mask[idx],
            self.xyz_feature[idx],
            self.labels[idx]
        )

# 训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature, labels) in enumerate(dataloader):
        # 将数据移动到设备上
        atom_list = atom_list.to(device)
        bond_list = bond_list.to(device)
        atom_degree_list = atom_degree_list.to(device)
        bond_degree_list = bond_degree_list.to(device)
        atom_mask = atom_mask.to(device)
        xyz_feature = xyz_feature.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# 验证函数
def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature, labels) in enumerate(dataloader):
            # 将数据移动到设备上
            atom_list = atom_list.to(device)
            bond_list = bond_list.to(device)
            atom_degree_list = atom_degree_list.to(device)
            bond_degree_list = bond_degree_list.to(device)
            atom_mask = atom_mask.to(device)
            xyz_feature = xyz_feature.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设你已经加载了数据
    # 这里是一个示例，你需要根据实际情况修改
    # 生成示例数据
    num_samples = 1000
    max_atoms = 100
    max_bonds = 200
    atom_feature_dim = 75
    bond_feature_dim = 10
    xyz_feature_dim = 5

    # 随机生成示例数据
    atom_list = torch.randn(num_samples, max_atoms, atom_feature_dim)
    bond_list = torch.randn(num_samples, max_bonds, bond_feature_dim)
    atom_degree_list = torch.randint(0, max_atoms, (num_samples, max_atoms))
    bond_degree_list = torch.randint(0, max_bonds, (num_samples, max_atoms))
    atom_mask = torch.randint(0, 2, (num_samples, max_atoms, 1))
    xyz_feature = torch.randn(num_samples, xyz_feature_dim, max_atoms)
    labels = torch.randn(num_samples, 1)  # 回归任务的标签

    # 创建数据集和数据加载器
    dataset = MolecularDataset(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, xyz_feature, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = PointGAT(radius=3, T=3, input_feature_dim=75, input_bond_dim=10, fingerprint_dim=128, output_units_num=1, p_dropout=0.2, xyz_feature_dim=5)
    model = model.to(device)
    criterion = nn.MSELoss()  # 如果是回归任务
    # criterion = nn.CrossEntropyLoss()  # 如果是分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 100
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    print("Training completed!")

if __name__ == "__main__":
    main()
