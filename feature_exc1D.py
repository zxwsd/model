import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import Normalizer
import os

# 将SMILES字符串转换为1024维度的Morgan分子指纹向量
def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fp_str = fp.ToBitString()
        return [int(bit) for bit in fp_str]
    else:
        return [0] * nBits

# 提取13C表的shifts列并进行等距采样
def extract_13Cshifts(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    grouped_shifts = df.groupby('ID')['shifts'].apply(list).reset_index()
    result = {}
    for index, row in grouped_shifts.iterrows():
        shifts = row['shifts']
        shifts_array = np.array(shifts)
        new_shifts = np.arange(0, 220, 0.1)
        binary_vector = np.zeros(2200)
        mask = np.isin(new_shifts, shifts_array)
        binary_vector[mask] = 1
        result[row['ID']] = binary_vector
    return result

# 提取1H表的shifts列并进行等距采样
def extract_1Hshifts(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    grouped_data = df.groupby('ID')[['shifts', 'H_num']].apply(lambda x: pd.Series({'shifts': x['shifts'].values, 'H_num': x['H_num'].values})).reset_index()
    result = {}
    for index, row in grouped_data.iterrows():
        shifts = row['shifts']
        h_nums = row['H_num']
        new_shifts = np.arange(0, 17, 0.01)
        sampled_vector = np.zeros(1700)
        for shift, h_num in zip(shifts, h_nums):
            idx = int((shift - 0) / 0.01)
            sampled_vector[idx] = h_num
        result[row['ID']] = sampled_vector
    return result

# 提取LCMS表的MZ列并进行等距采样
def extract_LCMS_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    grouped_data = df.groupby('ID')[['MZ', 'Abundence']].apply(lambda x: pd.Series({'MZ': x['MZ'].values, 'Abundence': x['Abundence'].values})).reset_index()
    result = {}
    for index, row in grouped_data.iterrows():
        shifts = row['MZ']
        h_nums = row['Abundence']
        new_shifts = np.arange(0, 500, 0.1)
        sampled_vector = np.zeros(5000)
        for shift, h_num in zip(shifts, h_nums):
            idx = int((shift - 0) / 0.1)
            sampled_vector[idx] = h_num
        result[row['ID']] = sampled_vector
    return result

# 拼接四个函数生成的一维向量并保存为单独的.npy文件
def concatenate_and_save_vectors(file_path, output_dir):
    # 读取SMILES列并生成分子指纹向量
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    if 'SMILES' not in df.columns:
        raise ValueError("Excel文件中没有找到SMILES列。")
    fingerprints = df['SMILES'].apply(lambda x: smiles_to_fingerprint(x)).tolist()

    # 提取13C、1H和LCMS数据
    extract_13C = extract_13Cshifts(file_path, '13C')
    extract_1H = extract_1Hshifts(file_path, '1H')
    extract_LCMS = extract_LCMS_data(file_path, 'LCMS')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 拼接向量并逐个保存
    i=0
    for idx, id in enumerate(extract_13C.keys()):
        vector_13C = extract_13C.get(id, np.zeros(2200))
        vector_1H = extract_1H.get(id, np.zeros(1700))
        vector_LCMS = extract_LCMS.get(id, np.zeros(5000))
        vector_fp = fingerprints[idx] if idx < len(fingerprints) else np.zeros(1024)

        # 拼接四个特征向量
        concatenated_vector = np.concatenate([np.array(vector_fp), vector_13C, vector_1H, vector_LCMS])

        # 保存为单独的.npy文件
        
        np.save(f'{output_dir}/one_dim_{i}.npy', concatenated_vector )
        i+=1
        print(f'保存了化合物 {id} 的特征向量：{output_dir}/one_dim_{i}.npy' )

# 使用函数
file_path = 'data/dataset.xlsx'
output_dir = 'data/feature/1D'
concatenate_and_save_vectors(file_path, output_dir)
