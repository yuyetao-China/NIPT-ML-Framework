# 双阶段概率随机森林：患病概率 + 预测可靠性
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")
from config import FEMALE_DIR, RESULTS_DIR

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 复现 DSE-TV 方法（论文Li et al.）
class DSE_TV:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.threshold = 0.3
    def fit(self, X, y):
        self.model1 = XGBClassifier(n_estimators=100, random_state=self.random_state)
        self.model2 = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.model1.fit(X, y); self.model2.fit(X, y)
        self.weights = (0.8, 0.2)
        return self
    def predict(self, X):
        p = self.weights[0] * self.model1.predict_proba(X)[:,1] + self.weights[1] * self.model2.predict_proba(X)[:,1]
        return (p >= self.threshold).astype(int)

def bootstrap_two_stage(X_A, X_B, y, n_iter=1000):
    """Bootstrap评估双阶段模型的泛化稳定性"""
    f1_scores = []
    np.random.seed(42)
    for _ in range(n_iter):
        idx = np.random.choice(len(y), len(y), replace=True)
        XA_b, XB_b = X_A.iloc[idx], X_B.iloc[idx]
        y_b = y[idx]
        rf_a = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_a.fit(XA_b, y_b)
        predA = rf_a.predict(XA_b)
        isCorrect = (predA == y_b).astype(int)
        rf_b = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_b.fit(XB_b, isCorrect)
        probA = rf_a.predict_proba(X_A)[:,1]
        probB = rf_b.predict_proba(X_B)[:,1]
        final = (probA * probB + (1-probA)*(1-probB) > 0.5).astype(int)
        f1_scores.append(f1_score(y, final))
    return np.mean(f1_scores), np.std(f1_scores), np.percentile(f1_scores, [2.5,97.5])

def run_full_analysis(file_path, disease_name):
    print(f"\n{'='*60}\nAnalyzing {disease_name}\n{'='*60}")
    df = pd.read_excel(file_path, sheet_name='女胎检测数据')
    df.columns = df.columns.str.replace('\n','').str.strip()
    y = df['患病检测准确率'].values

    # 特征集A（染色体信号）
    feat_A = ['孕妇BMI', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
    X_A = df[feat_A].copy()
    # 特征集B（测序质量）
    feat_B = ['检测抽血次数','检测孕周','孕妇BMI','原始读段数','在参考基因组上比对的比例',
              '重复读段的比例','唯一比对的读段数','GC含量','X染色体浓度','13号染色体的GC含量',
              '18号染色体的GC含量','21号染色体的GC含量','被过滤掉读段数的比例','NIPT检测准确性']
    X_B = df[feat_B].copy()

    # 删除缺失值
    valid = X_A.notna().all(axis=1) & X_B.notna().all(axis=1) & ~np.isnan(y)
    X_A, X_B, y = X_A[valid], X_B[valid], y[valid]
    print(f"有效样本: {len(y)} (患病:{np.sum(y)}, 健康:{len(y)-np.sum(y)})")

    # Bootstrap
    bs_mean, bs_std, (bs_low, bs_high) = bootstrap_two_stage(X_A, X_B, y)
    print(f"Bootstrap F1: mean={bs_mean:.4f}, std={bs_std:.4f}, 95%CI=[{bs_low:.4f},{bs_high:.4f}]")

    # 10折交叉验证消融实验
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    ablation = {'仅模型A':[], '模型A+SMOTE':[], '完整双阶段':[]}
    for train_idx, test_idx in skf.split(X_A, y):
        XA_tr, XA_te = X_A.iloc[train_idx], X_A.iloc[test_idx]
        XB_tr, XB_te = X_B.iloc[train_idx], X_B.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 仅模型A
        rf_only = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_only.fit(XA_tr, y_tr)
        ablation['仅模型A'].append(f1_score(y_te, rf_only.predict(XA_te)))

        # 模型A + SMOTE
        sm = SMOTE(random_state=42)
        XA_sm, y_sm = sm.fit_resample(XA_tr, y_tr)
        rf_sm = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_sm.fit(XA_sm, y_sm)
        ablation['模型A+SMOTE'].append(f1_score(y_te, rf_sm.predict(XA_te)))

        # 完整双阶段
        rf_a = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_a.fit(XA_tr, y_tr)
        predA_tr = rf_a.predict(XA_tr)
        isCorrect = (predA_tr == y_tr).astype(int)
        rf_b = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_b.fit(XB_tr, isCorrect)
        probA = rf_a.predict_proba(XA_te)[:,1]
        probB = rf_b.predict_proba(XB_te)[:,1]
        final = (probA * probB + (1-probA)*(1-probB) > 0.5).astype(int)
        ablation['完整双阶段'].append(f1_score(y_te, final))

    print("\n消融实验F1 (10折):")
    for k, v in ablation.items():
        print(f"{k}: mean={np.mean(v):.4f} ± {np.std(v):.4f}")

    # 全量训练最终模型
    rf_a_full = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, oob_score=True)
    rf_a_full.fit(X_A, y)
    probA_full = rf_a_full.predict_proba(X_A)[:,1]
    predA_full = rf_a_full.predict(X_A)
    isCorrect_full = (predA_full == y).astype(int)
    rf_b_full = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf_b_full.fit(X_B, isCorrect_full)
    probB_full = rf_b_full.predict_proba(X_B)[:,1]
    final_pred = (probA_full * probB_full + (1-probA_full)*(1-probB_full) > 0.5).astype(int)

    acc = accuracy_score(y, final_pred)
    prec = precision_score(y, final_pred)
    rec = recall_score(y, final_pred)
    f1 = f1_score(y, final_pred)
    print(f"\n最终模型: Acc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}, F1={f1:.2%}")

    # 特征重要性
    imp = rf_a_full.feature_importances_
    print("\n模型A特征重要性:")
    for name, val in sorted(zip(feat_A, imp), key=lambda x: -x[1]):
        print(f"  {name}: {val:.4f}")

    # SHAP分析（采样100个样本）
    X_sample = X_A.sample(min(100, len(X_A)), random_state=42)
    explainer = shap.Explainer(rf_a_full, X_sample, algorithm='tree')
    shap_values = explainer.shap_values(X_sample)[1]   # 取正类
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_A, show=False)
    plt.title(f'{disease_name} SHAP Global Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/{disease_name}_SHAP.png', dpi=300)

    # 混淆矩阵
    cm = confusion_matrix(y, final_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['健康','患病'])
    disp.plot(cmap='Blues')
    plt.title(f'{disease_name} Confusion Matrix')
    plt.savefig(f'{RESULTS_DIR}/{disease_name}_cm.png', dpi=300)

    # 保存结果
    summary = pd.DataFrame({
        '指标': ['准确率','精确率','召回率','F1分数'],
        '数值': [acc, prec, rec, f1]
    })
    with pd.ExcelWriter(f'{RESULTS_DIR}/{disease_name}_results.xlsx') as writer:
        pd.DataFrame(ablation).to_excel(writer, sheet_name='消融实验', index=False)
        summary.to_excel(writer, sheet_name='本模型性能', index=False)
        pd.DataFrame({'特征':feat_A, '重要性':imp}).sort_values('重要性',ascending=False).to_excel(writer, sheet_name='特征重要性', index=False)
    return {'疾病':disease_name, '样本数':len(y), '患病数':np.sum(y), '准确率':acc, 'F1':f1}

if __name__ == '__main__':
    files = {
        'T13': f'{FEMALE_DIR}/T13_data.xlsx',
        'T18': f'{FEMALE_DIR}/T18_data.xlsx',
        'T21': f'{FEMALE_DIR}/T21_data.xlsx',
    }
    all_metrics = []
    for name, path in files.items():
        res = run_full_analysis(path, name)
        all_metrics.append(res)
    pd.DataFrame(all_metrics).to_excel(f'{RESULTS_DIR}/ThreeDiseases_summary.xlsx', index=False)
    print("\n所有分析完成，结果保存在 results/ 目录下。")
