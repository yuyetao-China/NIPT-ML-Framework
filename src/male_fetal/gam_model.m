% 广义加性模型拟合Y染色体浓度~孕周+BMI
clearvars; close all; clc;
warning off;
cfg = config();                                 % 加载路径配置
dataPath = fullfile(cfg.male_dir, 'male_data_gam.xlsx');
data = readtable(dataPath, 'Sheet', 'Sheet1');
data.Properties.VariableNames = {'PatientID', 'GestationalWeeks', 'BMI', 'YChromosomeConcentration'};
data(any(ismissing(data), 2), :) = [];         % 删除缺失行

fprintf('Y染色体浓度范围: [%.6f, %.6f]\n', min(data.YChromosomeConcentration), max(data.YChromosomeConcentration));
fprintf('孕周范围: [%.1f, %.1f] 周\n', min(data.GestationalWeeks), max(data.GestationalWeeks));
fprintf('BMI范围: [%.1f, %.1f]\n', min(data.BMI), max(data.BMI));

% 探索性数据可视化
figure('Position', [100, 100, 1200, 500], 'Color', 'w');
subplot(1,2,1);
scatter(data.GestationalWeeks, data.YChromosomeConcentration, 40, data.BMI, 'filled');
colorbar;
xlabel('孕周');
ylabel('Y染色体浓度');
title('Y浓度 vs 孕周 (按BMI着色)');
grid on;
subplot(1,2,2);
scatter(data.BMI, data.YChromosomeConcentration, 40, data.GestationalWeeks, 'filled');
colorbar;
xlabel('BMI');
ylabel('Y染色体浓度');
title('Y浓度 vs BMI (按孕周着色)');
grid on;
sgtitle('探索性数据分析', 'FontSize', 16, 'FontWeight', 'bold');

% 拟合广义加性模型（含交互项）
gamModel = fitrgam(data, 'YChromosomeConcentration ~ 1 + GestationalWeeks + BMI', ...
    'Interactions', 'all', ...
    'FitStandardDeviation', true, ...
    'Verbose', 1);
disp(gamModel);

% 预测并计算R²
ypred = predict(gamModel, data);
yactual = data.YChromosomeConcentration;
ss_res = sum((yactual - ypred).^2);
ss_tot = sum((yactual - mean(yactual)).^2);
r_squared = 1 - (ss_res / ss_tot);
fprintf('模型R²: %.4f\n', r_squared);

% 尝试输出模型参数详细信息
try
    modelInfo = gamModel;
    if isprop(modelInfo, 'ModelParameters')
        params = modelInfo.ModelParameters;
        paramNames = properties(params);
        for i = 1:length(paramNames)
            paramName = paramNames{i};
            paramValue = params.(paramName);
            if isnumeric(paramValue) || ischar(paramValue) || islogical(paramValue)
                if isnumeric(paramValue) && numel(paramValue) == 1
                    paramNameCN = getParamNameCN(paramName);
                    fprintf('• %s (%s): %.4f\n', paramNameCN, paramName, paramValue);
                elseif ischar(paramValue)
                    paramNameCN = getParamNameCN(paramName);
                    fprintf('• %s (%s): %s\n', paramNameCN, paramName, paramValue);
                elseif islogical(paramValue)
                    paramNameCN = getParamNameCN(paramName);
                    fprintf('• %s (%s): %s\n', paramNameCN, paramName, string(paramValue));
                end
            end
        end
    else
        fprintf('• 模型类型: %s\n', class(gamModel));
        if isprop(gamModel, 'NumTrees')
            fprintf('• 树数量: %d\n', gamModel.NumTrees);
        end
        if isprop(gamModel, 'InteractionDepth')
            fprintf('• 交互深度: %d\n', gamModel.InteractionDepth);
        end
        if isprop(gamModel, 'MinLeafSize')
            fprintf('• 最小叶子大小: %d\n', gamModel.MinLeafSize);
        end
    end
catch ME
    fprintf('• 无法获取模型参数详细信息: %s\n', ME.message);
end
fprintf('• 截距: %.6f\n', gamModel.Intercept);
fprintf('• R²: %.4f\n', r_squared);

function paramNameCN = getParamNameCN(paramNameEN)
    paramMap = containers.Map(...
        {'NumPrint', 'MaxPValue', 'InitialLearnRateForPredictors', ...
         'InitialLearnRateForInteractions', 'NumTreesPerPredictor', ...
         'NumTreesPerInteraction', 'MaxNumSplitsPerPredictor', ...
         'MaxNumSplitsPerInteraction', 'VerbosityLevel', 'Interactions', ...
         'Version', 'Method', 'Type'}, ...
        {'打印次数', '最大P值', '预测变量初始学习率', ...
         '交互项初始学习率', '每个预测变量的树数量', ...
         '每个交互项的树数量', '每个预测变量的最大分裂数', ...
         '每个交互项的最大分裂数', '详细级别', '交互项', ...
         '版本', '方法', '类型'} ...
    );
    if isKey(paramMap, paramNameEN)
        paramNameCN = paramMap(paramNameEN);
    else
        paramNameCN = paramNameEN;
    end
end

% 生成预测曲面网格
weeks_range = linspace(min(data.GestationalWeeks), max(data.GestationalWeeks), 50);
bmi_range = linspace(min(data.BMI), max(data.BMI), 50);
[W, B] = meshgrid(weeks_range, bmi_range);
predData = table();
predData.GestationalWeeks = W(:);
predData.BMI = B(:);
Z = predict(gamModel, predData);
Z = reshape(Z, size(W));

% 曲面图 + 交互效应切片
figure('Position', [100, 100, 1400, 600], 'Color', 'w');
subplot(1,2,1);
surf(W, B, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
xlabel('孕周');
ylabel('BMI');
zlabel('Y染色体浓度');
title('GAM预测曲面');
colorbar;
grid on;
view(-45, 30);
subplot(1,2,2);
hold on;
bmi_levels = prctile(data.BMI, [10, 50, 90]);
colors = lines(length(bmi_levels));
for i = 1:length(bmi_levels)
    [~, idx] = min(abs(bmi_range - bmi_levels(i)));
    plot(weeks_range, Z(idx, :), 'LineWidth', 2, 'Color', colors(i, :), ...
        'DisplayName', sprintf('BMI=%.1f', bmi_range(idx)));
end
xlabel('孕周');
ylabel('预测Y染色体浓度');
title('不同BMI水平下的交互效应');
legend('show', 'Location', 'best');
grid on;
sgtitle('GAM模型：Y染色体浓度 vs 孕周和BMI', ...
        'FontSize', 16, 'FontWeight', 'bold');

% 边际效应图
figure('Position', [100, 100, 1200, 500], 'Color', 'w');
subplot(1,2,1);
plotPartialDependence(gamModel, 'GestationalWeeks');
grid on;
title('孕周的边际效应');
xlabel('孕周');
ylabel('Y染色体浓度');
subplot(1,2,2);
plotPartialDependence(gamModel, 'BMI');
grid on;
title('BMI的边际效应');
xlabel('BMI');
ylabel('Y染色体浓度');
sgtitle('预测变量的边际效应', 'FontSize', 16, 'FontWeight', 'bold');

% 交互热图 + 4%浓度等值线
figure('Position', [100, 100, 800, 600], 'Color', 'w');
n_colors = 256;
custom_cmap = [linspace(1, 0.2, n_colors)', linspace(1, 0.2, n_colors)', linspace(1, 0.6, n_colors)'];
imagesc(weeks_range, bmi_range, Z);
set(gca, 'YDir', 'normal');
colormap(custom_cmap);
colorbar;
hold on;
[C, h] = contour(W, B, Z, 10, 'LineColor', 'k', 'ShowText', 'on');
clabel(C, h, 'FontSize', 8);
threshold = (min(Z(:)) + max(Z(:))) / 2;
hText = findobj(gca, 'Type', 'text');
for i = 1:length(hText)
    if contains(get(hText(i), 'Tag'), 'ContourLabel')
        z_val = str2double(get(hText(i), 'String'));
        if z_val > threshold
            set(hText(i), 'Color', 'white', 'FontWeight', 'bold');
        else
            set(hText(i), 'Color', 'black', 'FontWeight', 'bold');
        end
    end
end
xlabel('孕周');
ylabel('BMI');
title('交互热图：Y染色体浓度');
hold on;
[C4, h4] = contour(W, B, Z, [0.04, 0.04], 'LineColor', 'r', 'LineWidth', 2, ...
    'ShowText', 'on', 'LabelFormat', '%0.2f');
hText4 = findobj(gca, 'Type', 'text');
for i = 1:length(hText4)
    if contains(get(hText4(i), 'String'), '0.04')
        set(hText4(i), 'Color', 'white', 'BackgroundColor', 'red', ...
            'FontWeight', 'bold', 'EdgeColor', 'black');
    end
end

% 残差分析
residuals = yactual - ypred;
figure('Position', [100, 100, 1200, 400], 'Color', 'w');
subplot(1,3,1);
scatter(ypred, residuals, 40, 'b', 'filled');
hold on;
plot(xlim, [0, 0], 'r--', 'LineWidth', 2);
xlabel('预测值');
ylabel('残差');
title('残差 vs 预测值');
grid on;
subplot(1,3,2);
histogram(residuals, 20, 'FaceColor', 'b', 'EdgeColor', 'w');
xlabel('残差');
ylabel('频数');
title('残差分布');
grid on;
subplot(1,3,3);
qqplot(residuals);
title('残差Q-Q图');
grid on;
sgtitle('残差分析', 'FontSize', 16, 'FontWeight', 'bold');

% 模型显著性F检验
n = height(data);
p = 3;
ss_reg = ss_tot - ss_res;
ms_reg = ss_reg / p;
ms_res = ss_res / (n - p - 1);
F_statistic = ms_reg / ms_res;
p_value = 1 - fcdf(F_statistic, p, n - p - 1);
alpha = 0.05;
if p_value < alpha
    fprintf('在显著性水平α=%.3f下，模型整体显著（p < α）\n', alpha);
else
    fprintf('在显著性水平α=%.3f下，模型整体不显著（p ≥ α）\n', alpha);
end
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1);
fprintf('调整后R²: %.4f\n', adj_r_squared);

% 基于固定BMI分组（粗略）给出推荐孕周
bmi_groups = [0, 20, 28, 32, 36, 40, Inf];
group_names = {'<20', '20-28', '28-32', '32-36', '36-40', '>40'};
target_concentration = 0.04;
fprintf('各BMI组达到Y染色体浓度≥4%%所需的最小孕周:\n');
for i = 1:length(bmi_groups)-1
    bmi_low = bmi_groups(i);
    bmi_high = bmi_groups(i+1);
    group_data = data(data.BMI >= bmi_low & data.BMI < bmi_high, :);
    if height(group_data) > 0
        valid_weeks = group_data.GestationalWeeks(group_data.YChromosomeConcentration >= target_concentration);
        if ~isempty(valid_weeks)
            min_week = min(valid_weeks);
            fprintf('BMI组 %s: 推荐在 %.1f 周后检测\n', group_names{i}, min_week);
        else
            pred_bmi = (bmi_low + bmi_high) / 2;
            pred_weeks = linspace(min(data.GestationalWeeks), max(data.GestationalWeeks), 100)';
            pred_table = table(pred_weeks, repmat(pred_bmi, length(pred_weeks), 1), ...
                'VariableNames', {'GestationalWeeks', 'BMI'});
            pred_conc = predict(gamModel, pred_table);
            above_target = find(pred_conc >= target_concentration, 1);
            if ~isempty(above_target)
                fprintf('BMI组 %s: 模型预测推荐在 %.1f 周后检测\n', group_names{i}, pred_weeks(above_target));
            else
                fprintf('BMI组 %s: 在当前孕周范围内无法达到目标浓度\n', group_names{i});
            end
        end
    else
        fprintf('BMI组 %s: 无数据\n', group_names{i});
    end
end

% 保存结果到results目录
data.PredictedY = ypred;
data.Residuals = residuals;
outputPath = fullfile(cfg.results_dir, 'NIPT_Analysis_Results.xlsx');
writetable(data, outputPath, 'Sheet', '分析结果');
saveas(gcf, fullfile(cfg.results_dir, 'NIPT_Residual_Analysis.png'));
