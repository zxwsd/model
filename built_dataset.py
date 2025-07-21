import pandas as pd

# 读取三个Excel文件，保留原始数据类型和精度
nmr_df = pd.read_excel('data/NMR.xlsx', dtype={'SMILES': str, 'ID': str})
score_df = pd.read_excel('data/score.xlsx', dtype={'SMILES': str, 'score': float})
lcms_df = pd.read_excel('data/LCMS.xlsx', dtype={'SMILES': str, 'ID': str})

# 确保所有文件都有SMILES列
if 'SMILES' in nmr_df.columns and 'SMILES' in score_df.columns and 'SMILES' in lcms_df.columns:
    # 创建一个字典，将SMILES作为键，ID作为值
    nmr_dict = nmr_df.set_index('SMILES')['ID'].to_dict()
    lcms_dict = lcms_df.set_index('SMILES')['ID'].to_dict()
    score_dict = score_df.set_index('SMILES')['score'].to_dict()

    # 找出三个文件共有的SMILES
    common_smiles = set(nmr_df['SMILES']).intersection(score_df['SMILES'], lcms_df['SMILES'])

    # 为共有的SMILES创建一个新的DataFrame
    result_df = pd.DataFrame(columns=['SMILES', 'ID_NMR', 'ID_LCMS', 'Score'])

    # 遍历共有的SMILES，并将对应的ID和Score添加到结果DataFrame中
    for smiles in common_smiles:
        if smiles in nmr_dict and smiles in lcms_dict and smiles in score_dict:
            temp_df = pd.DataFrame({
                'SMILES': [smiles],
                'ID_NMR': [nmr_dict[smiles]],
                'ID_LCMS': [lcms_dict[smiles]],
                'Score': [score_dict[smiles]]
            })
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 保存结果到新的Excel文件，保留原始精度
    with pd.ExcelWriter('data/common_SMILES_with_scores.xlsx', engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, float_format='%.16f')
else:
    print("One or more of the Excel files do not contain a 'SMILES' column.")
    
#根据ID提取相应的数据到dataset中
import pandas as pd

# 读取data/dataset.xlsx中的Sheet1
dataset_path = 'data/dataset.xlsx'
sheet1_df = pd.read_excel(dataset_path, sheet_name='Sheet1')

# 读取data/NMR.xlsx中的13C表
nmr_path = 'data/LCMS.xlsx'
nmr_13c_df = pd.read_excel(nmr_path, sheet_name='Sheet2')

# 确保ID_NMR和ID列都是字符串类型，以避免比较时的类型不匹配
sheet1_df['ID_LCMS'] = sheet1_df['ID_LCMS'].astype(str)
nmr_13c_df['ID'] = nmr_13c_df['ID'].astype(str)

# 创建一个空的DataFrame来存储匹配的行
matched_df = pd.DataFrame()

# 遍历dataset中的ID_NMR列，查找NMR文件中匹配的ID
for id_nmr in sheet1_df['ID_LCMS']:
    matched_rows = nmr_13c_df[nmr_13c_df['ID'] == id_nmr]
    matched_df = pd.concat([matched_df, matched_rows], ignore_index=True)

# 如果matched_df不为空，保存到新的Excel文件中
if not matched_df.empty:
    output_path = 'data/matched_data.xlsx'
    matched_df.to_excel(output_path, index=False)
    print(f'Matched data has been saved to {output_path}')
else:
    print('No matched data found.')
    
#  找LCMS和NMR相同的SMILES
import pandas as pd

# 读取LCMS数据
lcms_df = pd.read_excel('data/LCMS.xlsx', sheet_name='Sheet1')

# 读取NMR数据
nmr_df = pd.read_excel('data/NMR.xlsx', sheet_name='Sheet1')

# 确保两个DataFrame都有Name列
if 'Name' in lcms_df.columns and 'Name' in nmr_df.columns:
    # 将NMR数据的SMILES列和Name列合并为一个字典
    nmr_dict = nmr_df.set_index('Name')['SMILES'].to_dict()
    
    # 遍历LCMS数据的Name列，将对应的SMILES写入LCMS数据的SMILES列
    lcms_df['SMILES'] = lcms_df['Name'].map(nmr_dict).fillna('')
    
    # 保存更新后的LCMS数据到新的Excel文件
    lcms_df.to_excel('data/LCMS_updated.xlsx', index=False)
else:
    print("One or both of the Excel files do not contain a 'Name' column.")    
