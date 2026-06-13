# PERMANOVA检验男胎和女胎数据分布差异
import pandas as pd
import numpy as np
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MALE_DIR, FEMALE_DIR

# 读取数据
male = pd.read_excel(os.path.join(FEMALE_DIR, 'male_data_for_permanova.xlsx'))
female = pd.read_excel(os.path.join(FEMALE_DIR, 'female_data_for_permanova.xlsx'))

print("男胎数据列名:", male.columns.tolist())
print("女胎数据列名:", female.columns.tolist())

common_cols = male.columns.intersection(female.columns)
print("共有列:", common_cols.tolist())
male = male[common_cols]
female = female[common_cols]

# 只保留数值列
num_cols = male.select_dtypes(include=[np.number]).columns.tolist()
print("数值列:", num_cols)
male_num = male[num_cols]
female_num = female[num_cols]

combined = pd.concat([male_num, female_num], ignore_index=True)
groups = ['男胎'] * len(male_num) + ['女胎'] * len(female_num)

# 欧氏距离矩阵
scaler = StandardScaler()
scaled = scaler.fit_transform(combined)
dist_matrix = squareform(pdist(scaled, metric='euclidean'))
dm = DistanceMatrix(dist_matrix, ids=[str(i) for i in range(len(combined))])

# PERMANOVA检验（9999次置换）
result = permanova(dm, groups, permutations=9999)
print("\nPERMANOVA结果:")
print(f"统计量: {result['test statistic']}")
print(f"p值: {result['p-value']}")
if result['p-value'] > 0.05:
    print("结论: 两组数据无显著差异，可以用男胎数据训练模型预测女胎数据")
else:
    print("结论: 两组数据存在显著差异，建议独立构建女胎模型")
