# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 读取Excel文件
# df = pd.read_excel('data/dataset.xlsx', sheet_name='Sheet1')

# # 添加一个名为'index'的列，用于保存样本的索引
# df['index'] = df.index

# # 首先划分验证集（10条数据）
# valid_df = df.sample(n=10, random_state=42)
# remaining_df = df.drop(valid_df.index)

# # 然后划分训练集和测试集（测试集20条，训练集剩余）
# train_df, test_df = train_test_split(remaining_df, test_size=20, stratify=remaining_df['level'], random_state=42)

# # 保存验证集、训练集和测试集的索引、ID和对应的label
# valid_indices_with_id_and_label = valid_df[['index', 'ID', 'level']].values.tolist()
# train_indices_with_id_and_label = train_df[['index', 'ID', 'level']].values.tolist()
# test_indices_with_id_and_label = test_df[['index', 'ID', 'level']].values.tolist()

# # 将索引、ID和label保存到新的Excel文件中
# with pd.ExcelWriter('data/dataset_split1.xlsx') as writer:
#     pd.DataFrame(valid_indices_with_id_and_label, columns=['valid_index', 'ID', 'label']).to_excel(writer, sheet_name='ValidIndicesWithIdAndLabel', index=False)
#     pd.DataFrame(train_indices_with_id_and_label, columns=['train_index', 'ID', 'label']).to_excel(writer, sheet_name='TrainIndicesWithIdAndLabel', index=False)
#     pd.DataFrame(test_indices_with_id_and_label, columns=['test_index', 'ID', 'label']).to_excel(writer, sheet_name='TestIndicesWithIdAndLabel', index=False)


# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 读取Excel文件
# df = pd.read_excel('data/dataset.xlsx', sheet_name='Sheet1')

# # 添加一个名为'index'的列，用于保存样本的索引
# df['index'] = df.index

# # 假设有一个名为'ID'的列，包含样本的唯一标识符
# # 如果您的ID列有不同的名称，请将'ID'替换为您数据集中对应的列名
# train_df, test_df = train_test_split(df, test_size=0.2,  stratify=df['level'])
# # random_state=42,
# # 保存样本的索引、ID和对应的label
# train_indices_with_id_and_label = train_df[['index', 'ID', 'level']].values.tolist()
# test_indices_with_id_and_label = test_df[['index', 'ID', 'level']].values.tolist()

# # 将索引、ID和label保存到新的Excel文件中
# with pd.ExcelWriter('data/dataset_split.xlsx') as writer:
#     pd.DataFrame(train_indices_with_id_and_label, columns=['train_index', 'ID', 'label']).to_excel(writer, sheet_name='TrainIndicesWithIdAndLabel', index=False)
#     pd.DataFrame(test_indices_with_id_and_label, columns=['test_index', 'ID', 'label']).to_excel(writer, sheet_name='TestIndicesWithIdAndLabel', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split

# 读取Excel文件
df = pd.read_excel('data/dataset.xlsx', sheet_name='Sheet1')

# 添加一个名为'index'的列，用于保存样本的索引
df['index'] = df.index

# 从type=1的数据中均匀挑选10条作为验证集，确保level分布均匀
type_1_df = df[df['type'] == 1]

# 按level分组并均匀抽样
valid_df = pd.DataFrame()
for level, group in type_1_df.groupby('level'):
    # 计算每个level应抽取的样本数（尽量均匀）
    num_samples = max(1, len(group) // (len(type_1_df) // 10))
    valid_df = pd.concat([valid_df, group.sample(n=num_samples, random_state=42)])

# 如果验证集不足10条，补充剩余的样本
if len(valid_df) < 10:
    remaining_type_1 = type_1_df.drop(valid_df.index)
    valid_df = pd.concat([valid_df, remaining_type_1.sample(n=10 - len(valid_df), random_state=42)])

remaining_df = df.drop(valid_df.index)

# 从剩下的数据中挑选20条作为测试集，剩下的作为训练集
test_df = remaining_df.sample(n=10, random_state=42)
train_df = remaining_df.drop(test_df.index)

# 保存验证集、训练集和测试集的索引、ID和对应的label
valid_indices_with_id_and_label = valid_df[['index', 'ID', 'level']].values.tolist()
train_indices_with_id_and_label = train_df[['index', 'ID', 'level']].values.tolist()
test_indices_with_id_and_label = test_df[['index', 'ID', 'level']].values.tolist()

# 将索引、ID和label保存到新的Excel文件中
with pd.ExcelWriter('data/dataset_split.xlsx') as writer:
    pd.DataFrame(valid_indices_with_id_and_label, columns=['valid_index', 'ID', 'label']).to_excel(writer, sheet_name='ValidIndicesWithIdAndLabel', index=False)
    pd.DataFrame(train_indices_with_id_and_label, columns=['train_index', 'ID', 'label']).to_excel(writer, sheet_name='TrainIndicesWithIdAndLabel', index=False)
    pd.DataFrame(test_indices_with_id_and_label, columns=['test_index', 'ID', 'label']).to_excel(writer, sheet_name='TestIndicesWithIdAndLabel', index=False)