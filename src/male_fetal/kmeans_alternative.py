# 方案二：K-means++聚类（同时考虑BMI和最佳孕周）
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import sys

# 添加项目根目录到路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MALE_DIR, RESULTS_DIR

# 绘图字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

# 读取数据
file_path = os.path.join(MALE_DIR, 'male_kmeans_data.xlsx')
df = pd.read_excel(file_path)
df_clean = df.dropna(subset=['OptimalGestationalWeeks', '孕妇BMI'])
X = df_clean[['OptimalGestationalWeeks', '孕妇BMI']].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 轮廓系数法选择最佳聚类数
inertias = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2   # 索引从0开始，对应k=2
print(f"最佳聚类数量: {optimal_k}")

# 最终聚类
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clean['聚类'] = clusters

# 统计信息
bmi_summary = df_clean.groupby('聚类')['孕妇BMI'].agg(['min', 'max', 'mean', 'std'])
gestational_summary = df_clean.groupby('聚类')['OptimalGestationalWeeks'].agg(['min', 'max', 'mean', 'std'])
print("\n每个聚类的BMI统计信息：")
print(bmi_summary)
print("\n每个聚类的最佳检测孕周统计信息：")
print(gestational_summary)

# 散点图（聚类结果）
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_clean['OptimalGestationalWeeks'], df_clean['孕妇BMI'],
                      c=df_clean['聚类'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='聚类')
plt.xlabel('最佳检测孕周')
plt.ylabel('孕妇BMI')
plt.title(f'K-Means++ 聚类分析 (k={optimal_k}) - 孕妇BMI与最佳检测孕周的关系')
plt.grid(True)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, alpha=0.8, label='聚类中心')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_scatter.png'), dpi=300)
plt.show()

# 详细输出每个聚类
print("\n每个聚类的详细描述：")
for i in range(optimal_k):
    cluster_data = df_clean[df_clean['聚类'] == i]
    print(f"\n聚类 {i}:")
    print(f"  样本数量: {len(cluster_data)}")
    print(f"  BMI范围: {cluster_data['孕妇BMI'].min():.2f} - {cluster_data['孕妇BMI'].max():.2f}")
    print(f"  平均BMI: {cluster_data['孕妇BMI'].mean():.2f} ± {cluster_data['孕妇BMI'].std():.2f}")
    print(f"  最佳检测孕周范围: {cluster_data['OptimalGestationalWeeks'].min():.2f} - {cluster_data['OptimalGestationalWeeks'].max():.2f}")
    print(f"  平均最佳检测孕周: {cluster_data['OptimalGestationalWeeks'].mean():.2f} ± {cluster_data['OptimalGestationalWeeks'].std():.2f}")
    gestational_counts = cluster_data['OptimalGestationalWeeks'].value_counts().sort_index()
    print(f"  最佳检测孕周分布:")
    for week, count in gestational_counts.items():
        print(f"    {week}周: {count}人 ({count / len(cluster_data) * 100:.1f}%)")

# BMI 分布直方图，并标出各聚类边界
plt.figure(figsize=(12, 8))
plt.hist(df_clean['孕妇BMI'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
for i in range(optimal_k):
    cluster_data = df_clean[df_clean['聚类'] == i]
    bmi_min = cluster_data['孕妇BMI'].min()
    bmi_max = cluster_data['孕妇BMI'].max()
    color = colors[i % len(colors)]
    plt.axvline(x=bmi_min, color=color, linestyle='--', linewidth=2,
                label=f'聚类 {i} BMI范围: {bmi_min:.2f}-{bmi_max:.2f}')
    plt.axvline(x=bmi_max, color=color, linestyle='--', linewidth=2)
plt.xlabel('BMI', fontsize=16)
plt.ylabel('频数', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_bmi_histogram.png'), dpi=300)
plt.show()
