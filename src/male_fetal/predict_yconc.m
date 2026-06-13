% 使用混合效应模型预测Y染色体浓度
clear; clc;
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'male_data_for_lme.xlsx'), 'Sheet', 'Sheet1', 'VariableNamingRule','preserve');
data.Properties.VariableNames = {'SubjectID','Age','Height','Weight','BMI','GC_content',...
    'Gestational_week','Y_chromosome_concentration','Pregnancy_count','Delivery_count'};

% 缺失值用均值填充
numeric_vars = {'Age','Height','Weight','BMI','GC_content','Gestational_week','Y_chromosome_concentration','Pregnancy_count','Delivery_count'};
for i = 1:length(numeric_vars)
    col = numeric_vars{i};
    missing = ismissing(data.(col));
    if any(missing)
        data.(col)(missing) = mean(data.(col), 'omitnan');
    end
end

% 标准化连续变量（用于混合效应模型）
norm_vars = {'Gestational_week','BMI','Age','Height','Weight','GC_content','Pregnancy_count','Delivery_count'};
for i = 1:length(norm_vars)
    col = norm_vars{i};
    data.(['norm_',col]) = zscore(data.(col));
end

% 拟合线性混合效应模型（带随机截距）
formula = 'Y_chromosome_concentration ~ 1 + norm_Gestational_week^2 + norm_BMI^2 + norm_Age + norm_Height + norm_Weight + norm_GC_content + norm_Pregnancy_count + norm_Delivery_count + (1 | SubjectID)';
lme = fitlme(data, formula);
fprintf('R² = %.4f\n', 1 - sum((data.Y_chromosome_concentration - predict(lme,data)).^2) / sum((data.Y_chromosome_concentration - mean(data.Y_chromosome_concentration)).^2));

% 保存预测值
data.Predicted_Y = predict(lme, data);
writetable(data, fullfile(cfg.results_dir, 'Predicted_Y_Concentration.xlsx'));
