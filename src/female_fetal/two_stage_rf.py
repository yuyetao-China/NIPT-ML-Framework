# 双阶段概率随机森林：患病概率 + 预测可靠性
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from xgboost import XGBClassifier
import shap
import warnings
import os
import sys

# 添加项目根目录到路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FEMALE_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")

# 全局绘图设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.title_fontsize"] = 14


# DSE-TV 复现类（动态策略集成 + 阈值投票）
class DSE_TV:
    """动态策略集成 + 阈值投票 (复现原论文: NB:0.8 + DT:0.2, 阈值0.3)"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.threshold = 0.3

    def fit(self, X, y):
        self.model1 = XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=self.random_state
        )
        self.model2 = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state
        )
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        self.weights = (0.8, 0.2)
        return self

    def predict(self, X):
        p1 = self.model1.predict_proba(X)[:, 1]
        p2 = self.model2.predict_proba(X)[:, 1]
        p_fusion = self.weights[0] * p1 + self.weights[1] * p2
        return (p_fusion >= self.threshold).astype(int)


# Bootstrap 评估双阶段模型
def bootstrap_two_stage(X_A, X_B, y, n_iter=1000):
    """对完整双阶段模型进行 Bootstrap 重抽样，返回 F1 分数的均值、标准差、95% CI"""
    f1_scores = []
    np.random.seed(42)
    for _ in range(n_iter):
        idx = np.random.choice(len(y), len(y), replace=True)
        XA_boot, XB_boot = X_A.iloc[idx], X_B.iloc[idx]
        y_boot = y[idx]

        # 训练双阶段模型
        rf_a = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_a.fit(XA_boot, y_boot)
        predA = rf_a.predict(XA_boot)
        isCorrect = (predA == y_boot).astype(int)
        rf_b = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_b.fit(XB_boot, isCorrect)

        # 预测所有原始样本（用于评估泛化性能）
        probA = rf_a.predict_proba(X_A)[:, 1]
        probB = rf_b.predict_proba(X_B)[:, 1]
        final_pred = (probA * probB + (1 - probA) * (1 - probB) > 0.5).astype(int)
        f1_scores.append(f1_score(y, final_pred))

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    ci_low, ci_high = np.percentile(f1_scores, [2.5, 97.5])
    return mean_f1, std_f1, ci_low, ci_high


# 主分析函数
def run_full_analysis(file_path, disease_name):
    print(f"\n{'=' * 60}")
    print(f"正在分析疾病: {disease_name}")
    print(f"{'=' * 60}")

    # 读取数据
    df = pd.read_excel(file_path, sheet_name="女胎检测数据")
    df.columns = df.columns.str.replace("\n", "").str.strip()
    y = df["患病检测准确率"].values

    # 特征集A（染色体信号）
    features_A = [
        "孕妇BMI",
        "13号染色体的Z值",
        "18号染色体的Z值",
        "21号染色体的Z值",
        "X染色体的Z值",
    ]
    X_A = df[features_A].copy()

    # 特征集B（测序质量）
    features_B = [
        "检测抽血次数",
        "检测孕周",
        "孕妇BMI",
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "GC含量",
        "X染色体浓度",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "被过滤掉读段数的比例",
        "NIPT检测准确性",
    ]
    X_B = df[features_B].copy()

    # 删除缺失值
    valid = X_A.notna().all(axis=1) & X_B.notna().all(axis=1) & ~np.isnan(y)
    X_A = X_A[valid]
    X_B = X_B[valid]
    y = y[valid]
    print(f"有效样本数: {len(y)} (患病: {np.sum(y)}, 健康: {len(y) - np.sum(y)})")

    # Bootstrap 1000 次
    print("\n正在执行 Bootstrap 重抽样 (1000次)，计算双阶段模型 F1 的置信区间...")
    bs_mean, bs_std, bs_low, bs_high = bootstrap_two_stage(X_A, X_B, y, n_iter=1000)
    print(f"【Bootstrap (n=1000) 双阶段模型 F1】")
    print(f"均值: {bs_mean:.4f}, 标准差: {bs_std:.4f}")
    print(f"95% 置信区间: [{bs_low:.4f}, {bs_high:.4f}]")

    # 消融实验：10折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    ablation_results = {"仅模型A": [], "模型A+SMOTE": [], "完整双阶段": []}

    for train_idx, test_idx in skf.split(X_A, y):
        XA_train, XA_test = X_A.iloc[train_idx], X_A.iloc[test_idx]
        XB_train, XB_test = X_B.iloc[train_idx], X_B.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 仅模型A
        rf_only = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_only.fit(XA_train, y_train)
        y_pred_only = rf_only.predict(XA_test)
        f1_only = f1_score(y_test, y_pred_only)
        ablation_results["仅模型A"].append(f1_only)

        # 模型A + SMOTE
        sm = SMOTE(random_state=42)
        XA_sm, y_sm = sm.fit_resample(XA_train, y_train)
        rf_sm = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_sm.fit(XA_sm, y_sm)
        y_pred_sm = rf_sm.predict(XA_test)
        f1_sm = f1_score(y_test, y_pred_sm)
        ablation_results["模型A+SMOTE"].append(f1_sm)

        # 完整双阶段模型
        rf_a = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_a.fit(XA_train, y_train)
        predA_train = rf_a.predict(XA_train)
        isCorrect_train = (predA_train == y_train).astype(int)
        rf_b = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_b.fit(XB_train, isCorrect_train)
        probA_test = rf_a.predict_proba(XA_test)[:, 1]
        probB_test = rf_b.predict_proba(XB_test)[:, 1]
        P0 = probA_test * probB_test + (1 - probA_test) * (1 - probB_test)
        final_pred = (P0 > 0.5).astype(int)
        f1_full = f1_score(y_test, final_pred)
        ablation_results["完整双阶段"].append(f1_full)

    print("\n【消融实验 - 10折交叉验证 F1 分数】")
    ablation_summary = {}
    for cfg_name, scores in ablation_results.items():
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        print(f"{cfg_name}: 平均 F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        ablation_summary[cfg_name] = {"均值": mean_f1, "标准差": std_f1}

    # 全量数据训练最终模型（用于后续评估和可视化）
    rf_a_full = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=42, oob_score=True
    )
    rf_a_full.fit(X_A, y)
    predA_full = rf_a_full.predict(X_A)
    probA_full = rf_a_full.predict_proba(X_A)[:, 1]
    isCorrect_full = (predA_full == y).astype(int)

    rf_b_full = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=42
    )
    rf_b_full.fit(X_B, isCorrect_full)
    probB_full = rf_b_full.predict_proba(X_B)[:, 1]
    P0_full = probA_full * probB_full + (1 - probA_full) * (1 - probB_full)
    final_pred_full = (P0_full > 0.5).astype(int)

    final_acc = accuracy_score(y, final_pred_full)
    final_precision = precision_score(y, final_pred_full)
    final_recall = recall_score(y, final_pred_full)
    final_f1 = f1_score(y, final_pred_full)

    print(f"\n【全量最终模型性能】")
    print(f"准确率: {final_acc:.2%}")
    print(f"精确率: {final_precision:.2%}")
    print(f"召回率: {final_recall:.2%}")
    print(f"F1分数: {final_f1:.2%}")

    # 特征重要性
    impA = rf_a_full.feature_importances_
    idxA = np.argsort(impA)[::-1]
    print(f"\n【模型A特征重要性】")
    for i, idx in enumerate(idxA):
        print(f"{i + 1}. {features_A[idx]}: {impA[idx]:.4f}")

    # SHAP 分析
    try:
        X_A_sample = X_A.sample(min(100, len(X_A)), random_state=42)
        explainer = shap.Explainer(rf_a_full, X_A_sample, algorithm="tree")
        shap_values = explainer.shap_values(X_A_sample)
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values[:, :, 1]

        if shap_values_class1.shape[1] == X_A_sample.shape[1]:
            # SHAP 条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values_class1,
                X_A_sample,
                plot_type="bar",
                show=False,
                feature_names=features_A,
            )
            plt.title(f"{disease_name} SHAP全局特征重要性", fontweight="bold")
            plt.xlabel("平均|SHAP值|")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_SHAP_全局重要性.png"), dpi=300)
            plt.close()

            # SHAP 蜂群摘要图
            plt.figure(figsize=(14, 8))
            shap.summary_plot(
                shap_values_class1, X_A_sample, show=False, feature_names=features_A
            )
            plt.xlabel("SHAP值（对模型输出的影响）", fontweight="bold")
            plt.title(f"{disease_name} SHAP全局特征重要性摘要图", fontweight="bold")
            # 修改颜色条标签
            cb = plt.gcf().get_axes()
            for ax_obj in cb:
                if ax_obj.get_label() == "<colorbar>":
                    ax_obj.set_ylabel("特征值", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_SHAP_全局特征重要性摘要图.png"), dpi=300)
            plt.close()

            # 单个患病样本 SHAP 力图
            pos_idx = np.where(y == 1)[0]
            if len(pos_idx) > 0:
                sample_orig_idx = pos_idx[0]
                if sample_orig_idx not in X_A_sample.index:
                    X_A_sample = pd.concat(
                        [X_A_sample, X_A.iloc[[sample_orig_idx]]]
                    ).drop_duplicates()
                    shap_values = explainer.shap_values(X_A_sample)
                    if isinstance(shap_values, list):
                        shap_values_class1 = shap_values[1]
                    else:
                        shap_values_class1 = shap_values[:, :, 1]
                sample_idx_in_sampled = X_A_sample.index.get_loc(sample_orig_idx)
                sample_shap = shap_values_class1[
                    sample_idx_in_sampled : sample_idx_in_sampled + 1, :
                ]
                sample_data = X_A_sample.iloc[[sample_idx_in_sampled]]
                shap.force_plot(
                    explainer.expected_value[1],
                    sample_shap,
                    sample_data,
                    matplotlib=True,
                    show=False,
                )
                plt.title(f"{disease_name} 单个患病样本 SHAP 力图", fontweight="bold")
                plt.savefig(
                    os.path.join(RESULTS_DIR, f"{disease_name}_SHAP_力图.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                print("SHAP 维度不匹配，跳过绘图")
    except Exception as e:
        print(f"SHAP 分析出错（可忽略）: {e}")

    # 复现对比方法 (10折交叉验证)
    print("\n【复现对比方法并评估 (10折交叉验证)】")
    cv_results = {
        "DSE-TV": [],
        "XGBoost+SMOTE": [],
        "RUSBoost": [],
        "EasyEnsemble": [],
        "BalancedRandomForest": [],
        "我们的双阶段模型": [],
    }

    for train_idx, test_idx in skf.split(X_A, y):
        X_tr, X_te = X_A.iloc[train_idx], X_A.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # DSE-TV
        dse = DSE_TV(random_state=42)
        dse.fit(X_tr, y_tr)
        cv_results["DSE-TV"].append(f1_score(y_te, dse.predict(X_te)))

        # XGBoost + SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_tr, y_tr)
        xgb = XGBClassifier(n_estimators=100, random_state=42)
        xgb.fit(X_res, y_res)
        cv_results["XGBoost+SMOTE"].append(f1_score(y_te, xgb.predict(X_te)))

        # RUSBoost (使用 algorithm='SAMME')
        try:
            rus = RUSBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)
            rus.fit(X_tr, y_tr)
            cv_results["RUSBoost"].append(f1_score(y_te, rus.predict(X_te)))
        except (ValueError, RuntimeError):
            cv_results["RUSBoost"].append(np.nan)

        # EasyEnsemble
        ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
        ee.fit(X_tr, y_tr)
        cv_results["EasyEnsemble"].append(f1_score(y_te, ee.predict(X_te)))

        # BalancedRandomForest
        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
        brf.fit(X_tr, y_tr)
        cv_results["BalancedRandomForest"].append(f1_score(y_te, brf.predict(X_te)))

        # 我们的双阶段模型（使用同样的折）
        XB_tr = X_B.iloc[train_idx]
        XB_te = X_B.iloc[test_idx]
        rf_a_cv = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_a_cv.fit(X_tr, y_tr)
        predA_cv = rf_a_cv.predict(X_tr)
        isCorrect_cv = (predA_cv == y_tr).astype(int)
        rf_b_cv = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=42
        )
        rf_b_cv.fit(XB_tr, isCorrect_cv)
        probA_cv = rf_a_cv.predict_proba(X_te)[:, 1]
        probB_cv = rf_b_cv.predict_proba(XB_te)[:, 1]
        final_pred_cv = (
            probA_cv * probB_cv + (1 - probA_cv) * (1 - probB_cv) > 0.5
        ).astype(int)
        cv_results["我们的双阶段模型"].append(f1_score(y_te, final_pred_cv))

    print("\n【各模型 10 折交叉验证 F1 分数对比】")
    for name, scores in cv_results.items():
        valid_scores = [s for s in scores if not np.isnan(s)]
        if len(valid_scores) == len(scores):
            print(f"{name}: 平均 F1 = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        else:
            nan_count = len(scores) - len(valid_scores)
            print(f"{name}: 平均 F1 = {np.mean(valid_scores):.4f} ± {np.std(valid_scores):.4f}（{nan_count}/{len(scores)} 折失败）")

    # 交叉验证折线图（基于完整双阶段）
    folds = np.arange(1, 11)
    plt.figure(figsize=(8, 5))
    plt.plot(
        folds, ablation_results["完整双阶段"], "d-", linewidth=2, label="完整双阶段"
    )
    plt.xlabel("折次", fontweight="bold")
    plt.ylabel("F1 分数", fontweight="bold")
    plt.title(f"{disease_name} 10折交叉验证 F1 分数变化", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(folds)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_交叉验证折线图.png"), dpi=300)
    plt.close()

    # 混淆矩阵（使用全量最终模型）
    cm_final = confusion_matrix(y, final_pred_full)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_final, display_labels=["健康", "患病"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_xlabel("预测标签", fontweight="bold")
    ax.set_ylabel("真实标签", fontweight="bold")
    ax.set_title(f"{disease_name} 最终模型混淆矩阵", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_混淆矩阵.png"), dpi=300)
    plt.close()

    # 决策边界等高线图
    if len(features_A) >= 2:
        idx1, idx2 = idxA[0], idxA[1]
        feat1, feat2 = features_A[idx1], features_A[idx2]
        x_min, x_max = X_A[feat2].min() - 1, X_A[feat2].max() + 1
        y_min, y_max = X_A[feat1].min() - 1, X_A[feat1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        other_means = X_A.drop(columns=[feat1, feat2]).mean().values
        grid = np.zeros((xx.ravel().shape[0], X_A.shape[1]))
        grid[:, idx1] = yy.ravel()
        grid[:, idx2] = xx.ravel()
        col = 0
        for j in range(X_A.shape[1]):
            if j == idx1 or j == idx2:
                continue
            grid[:, j] = other_means[col]
            col += 1
        prob_grid = rf_a_full.predict_proba(grid)[:, 1].reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy, prob_grid, levels=20, cmap="RdBu_r", alpha=0.7)
        plt.colorbar(contour, ax=ax, label="患病概率")
        normal = X_A[y == 0]
        abnormal = X_A[y == 1]
        ax.scatter(
            normal[feat2],
            normal[feat1],
            c="blue",
            marker="o",
            edgecolors="k",
            label="健康样本",
        )
        ax.scatter(
            abnormal[feat2],
            abnormal[feat1],
            c="red",
            marker="s",
            edgecolors="k",
            label="异常样本",
        )
        ax.set_xlabel(feat2, fontweight="bold")
        ax.set_ylabel(feat1, fontweight="bold")
        ax.set_title(
            f"{disease_name} 决策边界 (基于{feat1}和{feat2})", fontweight="bold"
        )
        ax.legend()
        ax.contour(
            xx,
            yy,
            prob_grid,
            levels=[0.5],
            colors="black",
            linewidths=2,
            linestyles="--",
        )
        ax.annotate("阈值 0.5", xy=(xx.mean(), yy.mean()), fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_决策边界等高线图.png"), dpi=300)
        plt.close()

    # 部分依赖图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PartialDependenceDisplay.from_estimator(
        rf_a_full, X_A, [idxA[0], idxA[1]], ax=axes, grid_resolution=50
    )
    for ax in axes:
        ax.set_ylabel("部分依赖值", fontweight="bold")
    axes[0].set_title(
        f"{disease_name} 部分依赖: {features_A[idxA[0]]}", fontweight="bold"
    )
    axes[1].set_title(
        f"{disease_name} 部分依赖: {features_A[idxA[1]]}", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{disease_name}_部分依赖图.png"), dpi=300)
    plt.close()

    # 保存结果到 Excel
    with pd.ExcelWriter(
        os.path.join(RESULTS_DIR, f"{disease_name}_完整分析结果.xlsx"), engine="openpyxl"
    ) as writer:
        pd.DataFrame(ablation_results).to_excel(
            writer, sheet_name="消融实验_10折F1", index=False
        )
        pd.DataFrame(
            {
                "指标": ["准确率", "精确率", "召回率", "F1分数"],
                "数值": [final_acc, final_precision, final_recall, final_f1],
            }
        ).to_excel(writer, sheet_name="本模型性能", index=False)
        df_compare = pd.DataFrame(cv_results)
        df_compare.to_excel(writer, sheet_name="对比方法_10折F1", index=False)
        pd.DataFrame({"特征": features_A, "重要性": impA}).sort_values(
            "重要性", ascending=False
        ).to_excel(writer, sheet_name="特征重要性", index=False)
        pd.DataFrame(
            {
                "Bootstrap_F1均值": [bs_mean],
                "标准差": [bs_std],
                "95%置信区间下限": [bs_low],
                "95%置信区间上限": [bs_high],
            }
        ).to_excel(writer, sheet_name="Bootstrap结果", index=False)

    return {
        "疾病": disease_name,
        "样本数": len(y),
        "患病数": np.sum(y),
        "准确率": final_acc,
        "精确率": final_precision,
        "召回率": final_recall,
        "F1分数": final_f1,
        "Bootstrap_F1均值": bs_mean,
        "Bootstrap_F1_CI_low": bs_low,
        "Bootstrap_F1_CI_high": bs_high,
        "消融_仅模型A_F1": ablation_summary["仅模型A"]["均值"],
        "消融_模型A+SMOTE_F1": ablation_summary["模型A+SMOTE"]["均值"],
        "消融_完整双阶段_F1": ablation_summary["完整双阶段"]["均值"],
    }


# 主程序
if __name__ == "__main__":
    # 使用配置的路径读取三个疾病的Excel文件
    files = {
        "T13": os.path.join(FEMALE_DIR, "T13_data.xlsx"),
        "T18": os.path.join(FEMALE_DIR, "T18_data.xlsx"),
        "T21": os.path.join(FEMALE_DIR, "T21_data.xlsx"),
    }

    all_metrics = []
    for disease, path in files.items():
        res = run_full_analysis(path, disease)
        all_metrics.append(res)

    df_summary = pd.DataFrame(all_metrics)
    print("\n\n" + "=" * 80)
    print("三种疾病最终模型性能汇总")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    df_summary.to_excel(os.path.join(RESULTS_DIR, "三种疾病完整分析结果.xlsx"), sheet_name="性能汇总", index=False)
    print("\n所有结果已保存至 results 目录下的 Excel 文件。")
