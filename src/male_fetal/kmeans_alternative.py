# 方案二：K‑means++聚类（同时考虑BMI和最佳孕周）
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from config import MALE_DIR, RESULTS_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(MALE_DIR + '/问2聚类.xlsx')   # 用户自行准备
df_clean = df.dropna(subset=['OptimalGestationalWeeks', '孕妇BMI'])
X = df_clean[['OptimalGestationalWeeks', '孕妇BMI']].values
X_scaled = StandardScaler().fit_transform(X)

# 轮廓系数选择最佳k
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
optimal_k = K_range[sil_scores.index(max(sil_scores))]
print(f"最佳聚类数: {optimal_k}")

# 最终聚类
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['Cluster'] = clusters

# 输出每个聚类的BMI和孕周范围
summary = df_clean.groupby('Cluster').agg(
    BMI_min=('孕妇BMI','min'), BMI_max=('孕妇BMI','max'),
    Week_min=('OptimalGestationalWeeks','min'), Week_max=('OptimalGestationalWeeks','max')
)
print(summary)
summary.to_excel(RESULTS_DIR + '/kmeans_alternative_result.xlsx')
