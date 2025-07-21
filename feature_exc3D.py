import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_3d_coordinates(file_path, sheet_name):
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 确保SMILES列存在
    if 'SMILES' not in df.columns:
        raise ValueError("Excel文件中没有找到SMILES列。")
    
    # 初始化分子图数据列表
    molecular_graphs = []
    
    # 遍历每个SMILES字符串
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 为分子添加氢原子
            mol = Chem.AddHs(mol)
            
            # 生成三维坐标
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)  # 进行能量优化
            
            # 获取分子中的原子数
            num_atoms = mol.GetNumAtoms()
            # print(num_atoms)
            # 获取每个原子的三维坐标
            coordinates = mol.GetConformer().GetPositions()
            
            # 计算原子之间的相对位置
            relative_positions = np.zeros((num_atoms, num_atoms, 3))  # 创建N*N*3的零矩阵
            
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        relative_positions[i, j] = coordinates[i] - coordinates[j]  # 计算相对位置
            
            # 将当前分子的三维坐标和相对位置数据添加到列表中
            molecular_graphs.append((coordinates, relative_positions))
    
    return molecular_graphs

# 使用函数
file_path = 'data/dataset.xlsx'  # 请替换为你的文件路径
sheet_name = 'Sheet1'  # 请替换为你的工作表名称
molecular_graphs = smiles_to_3d_coordinates(file_path, sheet_name)

# 保存三维坐标和相对位置矩阵
for i, (coordinates, relative_positions) in enumerate(molecular_graphs):
    np.save(f'data/feature/3D/coordinates_molecule_{i}.npy', coordinates)
    np.save(f'data/feature/3D/relative_positions_molecule_{i}.npy', relative_positions)

# 打印保存的文件名
for i, (coordinates, relative_positions) in enumerate(molecular_graphs):
    print(f"Molecule {i+1} Coordinates saved as 'coordinates_molecule_{i}.npy'")
    print(f"Molecule {i+1} Relative Positions saved as 'relative_positions_molecule_{i}.npy'")
