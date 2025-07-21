import pandas as pd
import numpy as np

def extract_and_save_level(file_path, sheet_name, output_path):
    # 读取数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 提取 'level' 列
    level_data = df['level'].values  # 假设 'level' 列存在
    
    # 保存为 .npy 文件
    np.save(output_path, level_data)

# 使用示例
file_path = 'data/dataset.xlsx'  # 请替换为你的文件路径
sheet_name = 'Sheet1'  # 请替换为你的工作表名称
output_path = 'data/label.npy'  # 输出文件路径

# 提取并保存
extract_and_save_level(file_path, sheet_name, output_path)

print("数据已保存为：", output_path)
