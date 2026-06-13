% 使用混合效应模型预测Y染色体浓度
clear; clc;
cfg = config();                                 % 加载路径配置
data = readtable(fullfile(cfg.male_dir, 'male_data_for_lme.xlsx'), 'Sheet', 'Sheet1', 'VariableNamingRule','preserve');
data.Properties.VariableNames = {'SubjectID','Age','Height','Weight','BMI','GC_content',...
    'Gestational_week','Y_chromosome_concentration','Pregnancy_count','Delivery_count'};

fprintf('数据总行数: %d\n', height(data));
fprintf('缺失值统计:\n');
numeric_vars = {'Age','Height','Weight','BMI','GC_content','Gestational_week','Y_chromosome_concentration','Pregnancy_count','Delivery_count'};
for i = 1:length(numeric_vars)
    col = numeric_vars{i};
    missing = ismissing(data.(col));
    missing_count = sum(missing);
    fprintf('%s: %d 个缺失值\n', col, missing_count);
    if any(missing)
        col_mean = mean(data.(col), 'omitnan');
        data.(col)(missing) = col_mean;
    end
end

% 标准化连续变量（用于混合效应模型）
norm_vars = {'Gestational_week','BMI','Age','Height','Weight','GC_content','Pregnancy_count','Delivery_count'};
for i = 1:length(norm_vars)
    col = norm_vars{i};
    data.(['norm_', col]) = zscore(data.(col));
end

% 拟合线性混合效应模型（带随机截距）
formula = 'Y_chromosome_concentration ~ 1 + norm_Gestational_week^2 + norm_BMI^2 + norm_Age + norm_Height + norm_Weight + norm_GC_content + norm_Pregnancy_count + norm_Delivery_count + (1 | SubjectID)';
lme = fitlme(data, formula);
disp(lme);
fixed_effects = lme.Coefficients;
disp('固定效应系数:');
disp(fixed_effects);

% 预测并计算R²
y_pred = predict(lme, data);
y_actual = data.Y_chromosome_concentration;
SS_residual = sum((y_actual - y_pred).^2);
SS_total = sum((y_actual - mean(y_actual)).^2);
R_squared = 1 - (SS_residual / SS_total);
fprintf('模型拟合度 R²: %.4f\n', R_squared);

% 保存预测结果到results目录
data.Predicted_Y = y_pred;
output_filename = fullfile(cfg.results_dir, 'Predicted_Y_Concentration.xlsx');
writetable(data, output_filename);
disp(['预测结果已保存至：', output_filename]);
