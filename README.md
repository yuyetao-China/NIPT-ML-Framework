# NIPT-ML-Framework
Machine learning framework for NIPT: mixed-effects model for fetal DNA concentration prediction &amp; two-stage random forest for T13/T18/T21 chromosomal aneuploidy detection.

## 机器学习框架用于无创产前检测（NIPT）：
1. 男性胎儿：混合效应模型预测Y染色体浓度 + 个体化检测时机推荐
2. 女性胎儿：双阶段概率随机森林检测T13/T18/T21染色体异常

## 环境要求
1. MATLAB R2020b+ (Statistics and Machine Learning Toolbox)
2. Python 3.8+ (numpy, pandas, scikit-learn, imbalanced-learn, xgboost, shap, scikit-bio)

## 快速开始
1. 将原始数据放入 data/male/ 和 data/female/ 目录（见数据说明）
2. 复制 config/config_template.m 为 config.m（MATLAB）
3. 复制 config/config_template.py 为 config.py（Python）
4. 按顺序运行 src/male_fetal/ 中的MATLAB脚本，然后运行 src/female_fetal/two_stage_rf.py

## 结果
所有输出保存在 results/ 目录下
