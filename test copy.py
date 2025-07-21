import pandas as pd
from rdkit import Chem

def simplify_smiles(smiles_str):
    """简化 SMILES 字符串，去除立体异构等结构信息"""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None  # 无效的 SMILES 字符串
    simplified_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return simplified_smiles

def main():
    # 读取第一个文件：data/dataset.xlsx
    try:
        df_dataset = pd.read_excel("data/dataset.xlsx")
        print("已读取 data/dataset.xlsx")
    except Exception as e:
        print(f"无法读取 data/dataset.xlsx：{e}")
        return

    # 读取第二个文件：data/cb1_ligand_final.csv
    try:
        df_cb1 = pd.read_csv("data/cb1_ligand_final.csv")
        print("已读取 data/cb1_ligand_final.csv")
    except Exception as e:
        print(f"无法读取 data/cb1_ligand_final.csv：{e}")
        return

    # 确保两个文件中都有相应的列
    if "SMILES" not in df_dataset.columns:
        print("data/dataset.xlsx 中没有 'SMILES' 列！")
        return
    if "Ligand SMILES" not in df_cb1.columns:
        print("data/cb1_ligand_final.csv 中没有 'Ligand SMILES' 列！")
        return

    # 简化 SMILES
    df_dataset["Simplified SMILES"] = df_dataset["SMILES"]
    df_cb1["Simplified SMILES"] = df_cb1["Ligand SMILES"].apply(simplify_smiles)

    # 获取两列简化后的 SMILES
    simplified_smiles_dataset = df_dataset["Simplified SMILES"].dropna().tolist()
    simplified_smiles_cb1 = df_cb1["Simplified SMILES"].dropna().tolist()

    # 找出相同的简化 SMILES
    common_smiles = list(set(simplified_smiles_dataset) & set(simplified_smiles_cb1))

    # 输出到文件
    output_file = "data/common_smiles.txt"
    with open(output_file, "w") as f:
        for smi in common_smiles:
            f.write(f"{smi}\n")
    print(f"相同 SMILES 已保存到文件：{output_file}")

if __name__ == "__main__":
    main()