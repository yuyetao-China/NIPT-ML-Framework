# PERMANOVA检验男胎和女胎数据分布差异
import pandas as pd
import numpy as np
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from config import MALE_DIR, FEMALE_DIR

# 读取数据
male = pd.read_excel(MALE_DIR + '/male_female_common.xlsx', sheet_name='male')
female = pd.read_excel(FEMALE_DIR + '/male_female_common.xlsx', sheet_name='female')

common_cols = male.columns.intersection(female.columns)
male = male[common_cols]
female = female[common_cols]

# 只保留数值列
num_cols = male.select_dtypes(include=[np.number]).columns.tolist()
male_num = male[num_cols]
female_num = female[num_cols]

combined = pd.concat([male_num, female_num], ignore_index=True)
groups = ['male'] * len(male_num) + ['female'] * len(female_num)

# 欧氏距离矩阵
scaled = StandardScaler().fit_transform(combined)
dist_matrix = squareform(pdist(scaled, metric='euclidean'))
dm = DistanceMatrix(dist_matrix, ids=[str(i) for i in range(len(combined))])

# PERMANOVA检验（9999次置换）
result = permanova(dm, groups, permutations=9999)
print(f"PERMANOVA test statistic: {result['test statistic']}")
print(f"p-value: {result['p-value']}")
if result['p-value'] > 0.05:
    print("两组数据无显著差异，可以用男胎数据训练模型预测女胎")
else:
    print("两组数据存在显著差异，需独立构建女胎模型")
