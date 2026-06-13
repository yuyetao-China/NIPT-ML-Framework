% 广义加性模型拟合Y染色体浓度与孕周+BMI
clearvars; close all; clc;
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'male_data_gam.xlsx'), 'Sheet', 'Sheet1');
data.Properties.VariableNames = {'PatientID', 'GestationalWeeks', 'BMI', 'YChromosomeConcentration'};
data(any(ismissing(data),2), :) = [];

% 拟合GAM（含交互项）
gamModel = fitrgam(data, 'YChromosomeConcentration ~ 1 + GestationalWeeks + BMI', ...
    'Interactions', 'all', 'FitStandardDeviation', true);

% 预测与R²
y_pred = predict(gamModel, data);
y_actual = data.YChromosomeConcentration;
ss_res = sum((y_actual - y_pred).^2);
ss_tot = sum((y_actual - mean(y_actual)).^2);
r2 = 1 - ss_res/ss_tot;
fprintf('R² = %.4f\n', r2);

% 绘制预测曲面
weeks_range = linspace(min(data.GestationalWeeks), max(data.GestationalWeeks), 50);
bmi_range = linspace(min(data.BMI), max(data.BMI), 50);
[W, B] = meshgrid(weeks_range, bmi_range);
predData = table(W(:), B(:), 'VariableNames', {'GestationalWeeks','BMI'});
Z = predict(gamModel, predData); Z = reshape(Z, size(W));

figure; surf(W, B, Z, 'EdgeColor', 'none'); xlabel('Gestational Weeks'); ylabel('BMI');
zlabel('Y Concentration'); title('GAM Prediction Surface'); colorbar;

% 残差分析
residuals = y_actual - y_pred;
figure; subplot(1,3,1); scatter(y_pred, residuals); refline(0,0);
subplot(1,3,2); histogram(residuals); subplot(1,3,3); qqplot(residuals);

% 保存结果
data.PredictedY = y_pred; data.Residuals = residuals;
writetable(data, fullfile(cfg.results_dir, 'GAM_results.xlsx'));
