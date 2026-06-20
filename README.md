# NIPT-ML-Framework

基于混合效应模型和双阶段随机森林的胎儿DNA浓度预测与染色体异常检测

本仓库为NBU机器学习课程报告《基于混合效应模型和双阶段随机森林的胎儿DNA浓度预测与染色体异常检测》的**代码实现**。

---

## 🌟 亮点

- **任务协同优化**：将胎儿DNA浓度预测与染色体异常检测纳入统一框架，实现检测窗口推荐与异常识别的协同优化。
- **个体化建模**：利用广义加性模型和混合效应模型，刻画孕周、BMI、年龄等因素的非线性影响，浓度拟合优度 R² 从 0.364 提升至 0.805。
- **可靠性感知**：双阶段随机森林同时输出患病风险与预测置信度，为临床决策提供可量化的可靠性参考。
- **高召回率**：在 T13、T21 上召回率达 100%，T18 达 99.82%，F1 分数均高于 0.986。

---

## 📖 目录

- [数据介绍](#-数据介绍)
- [环境依赖](#-环境依赖)
- [代码文件说明](#-代码文件说明)
  - [男胎检测时机优化（MATLAB）](#男胎检测时机优化matlab)
  - [女胎染色体异常检测（Python）](#女胎染色体异常检测python)
- [运行说明](#-运行说明)
- [输出结果](#-输出结果)
- [引用](#-引用)
- [许可证](#-许可证)

---

## 📊 数据介绍

本研究使用的数据来源于 2025 年全国大学生数学建模竞赛 C 题，包含 **1082 例男胎** 和 **605 例女胎** 的无创产前检测（NIPT）脱敏临床记录。每个样本包含 24 个可用特征及对应的结局标签。

### 主要特征说明

| 特征类别 | 特征名称 | 说明 |
|---------|---------|------|
| 母体信息 | 孕周、BMI、年龄、怀孕次数、生产次数 | 用于个体化建模 |
| 染色体信号 | 13/18/21/X/Y 染色体的 Z 值 | 反映染色体剂量偏离程度 |
| 测序质量 | GC 含量、比对比例、重复读段比例、原始读段数等 | 反映测序过程质量 |
| 浓度指标 | Y 染色体浓度、X 染色体浓度 | 反映胎儿 DNA 浓度 |

### 数据用途

- **男胎数据**：利用 Y 染色体浓度作为胎儿 DNA 浓度的替代指标，建立预测模型并推荐个体化检测窗口。
- **女胎数据**：无 Y 染色体标记，利用染色体信号和测序质量特征构建异常检测模型，识别 T13、T18、T21 三种染色体异常。

---

## 🛠️ 环境依赖

### MATLAB

- **版本**：MATLAB R2020b 或更高版本
- **工具箱**：Statistics and Machine Learning Toolbox

### Python

- **版本**：Python 3.8 及以上
- **依赖安装**：

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost shap scikit-bio openpyxl
```

---

## 📂 代码文件说明

每个代码文件均可**独立运行**，完成特定的分析任务。

### 男胎检测时机优化（MATLAB）

所有脚本位于 `src/male_fetal/`，运行前请确保 MATLAB 当前文件夹在项目根目录。

| 文件名 | 功能说明 | 输入 | 输出 |
|--------|---------|------|------|
| `spearman_ttest.m` | Spearman 秩相关分析与 t 检验，筛选关键变量 | `male_data_full.xlsx` | 相关系数热图、t 检验结果 Excel |
| `gam_model.m` | 广义加性模型拟合 Y 染色体浓度 ~ 孕周 + BMI，含交互效应与可视化 | `male_data_gam.xlsx` | 预测曲面、边际效应图、热图、残差分析、R² |
| `bmi_clustering.m` | 基于 CH 指数的 K-means 聚类，自动确定 BMI 最优分组数 | `male_bmi_data.xlsx` | BMI 分组边界、聚类评估图 |
| `optimal_timing_plan1.m` | 方案一：基于 BMI 分组求解个体化最佳检测时点（含 Bootstrap 标准误） | `male_optimal.xlsx` | 各组最佳时点表、风险值图 |
| `multi_factor_grouping.m` | 三层分组辅助函数（怀孕次数 + 年龄 + BMI） | `multi_factor_data.xlsx` | 三层分组统计表 |
| `mixed_effect_model.m` | 混合效应模型拟合 + Bootstrap 标准误 + 高斯噪声敏感性分析 | `male_with_group.xlsx` | 固定效应系数、分组时点表、噪声敏感性图 |
| `predict_yconc.m` | 使用混合效应模型预测 Y 染色体浓度 | `male_data_for_lme.xlsx` | 预测值 Excel |

### 女胎染色体异常检测（Python）

所有脚本位于 `src/female_fetal/`。

| 文件名 | 功能说明 | 输入 | 输出 |
|--------|---------|------|------|
| `permanova_test.py` | PERMANOVA 检验男胎与女胎数据分布差异，验证迁移学习可行性 | `male_data_for_permanova.xlsx`、`female_data_for_permanova.xlsx` | PERMANOVA 统计量、p 值 |
| `two_stage_rf.py` | **核心脚本**：双阶段概率随机森林，包含完整训练、评估、可解释性分析 | `T13_data.xlsx`、`T18_data.xlsx`、`T21_data.xlsx` | 性能汇总 Excel、SHAP 图、混淆矩阵、决策边界图、部分依赖图 |

---

## 🚀 运行说明

每个代码文件均可独立运行，文件之间无依赖关系。你可以根据自己的需要，只运行某一个文件，也可以全部运行。

MATLAB 脚本
在 MATLAB 中，将当前文件夹切换到项目根目录（即包含 config.m 的文件夹），然后在命令窗口直接输入文件名运行。

| 想做什么 | 运行这个文件 |
|---------|-------------|
| 做 Spearman 相关分析和 t 检验，筛选关键变量 | `spearman_ttest` |
| 做 GAM 模型拟合，看孕周和 BMI 对 Y 染色体浓度的影响 | `gam_model` |
| 对 BMI 进行聚类，找最优分组数 | `bmi_clustering` |
| 方案一：计算每个 BMI 分组的最佳检测时点 | `optimal_timing_plan1` |
| 做三层分组（怀孕次数 + 年龄 + BMI）统计 | `multi_factor_grouping` |
| 做混合效应模型拟合 + 噪声敏感性分析 | `mixed_effect_model` |
| 用混合效应模型预测 Y 染色体浓度 | `predict_yconc` |

示例：

matlab
% 只想做 GAM 模型分析
gam_model

% 只想做混合效应模型
mixed_effect_model

Python 脚本
在终端中进入项目根目录，直接运行对应的 Python 文件。

想做什么	运行这个文件
做 PERMANOVA 检验，看男胎和女胎数据是否有显著差异	python src/female_fetal/permanova_test.py
做双阶段随机森林异常检测（T13/T18/T21）	python src/female_fetal/two_stage_rf.py
示例：

```bash
# 只想跑异常检测模型
python src/female_fetal/two_stage_rf.py
```

```bash
# 只想跑 PERMANOVA 检验
python src/female_fetal/permanova_test.py
```
两个 Python 文件也是相互独立的，想用哪个就运行哪个。

---

## 📊 输出结果

所有结果自动保存至 `results/` 目录：

| 输出类型 | 文件示例 |
|---------|----------|
| Excel 表格 | `T13_完整分析结果.xlsx`、`MixedEffect_Results.xlsx` |
| 可视化图表 | `T13_SHAP_全局重要性.png`、`T13_混淆矩阵.png`、`bmi_grouping_optimized_analysis.png` |
| 性能汇总 | `三种疾病完整分析结果.xlsx` |

---

## 📝 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{yu2026nipt,
  title={基于混合效应模型和双阶段随机森林的胎儿DNA浓度预测与染色体异常检测},
  author={俞烨涛 and 金施成 and 刘洁},
  journal={宁波大学课程设计报告},
  year={2026}
}
```

代码仓库：https://github.com/yuyetao-China/NIPT-ML-Framework

---

## 📄 许可证

MIT License
