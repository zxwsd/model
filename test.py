import pandas as pd
from rdkit import Chem

def simplify_smiles(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    simplified_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return simplified_smiles

def main():
    # 输入文件路径
    input_file = "data/cb1_ligand_final.xlsx"
    
    # 输出文件路径
    output_file = "data/cb1_ligand_simplified.csv"

    # 读取Excel文件
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"无法读取文件：{e}")
        return
    
    # 检查是否存在 "Ligand SMILES" 列
    if "Ligand SMILES" not in df.columns:
        print("文件中没有名为 'Ligand SMILES' 的列！")
        return
    
    # 简化 SMILES
    df["Simplified SMILES"] = df["Ligand SMILES"].apply(simplify_smiles)
    
    # 保存结果到新文件
    df.to_csv(output_file, index=False)
    
    print(f"简化后的 SMILES 已保存到文件：{output_file}")

if __name__ == "__main__":
    main()