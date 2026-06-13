% Spearman相关分析与t检验
clearvars; close all; clc;
cfg = config();                         % 加载路径配置
filename = fullfile(cfg.male_dir, 'male_data_full.xlsx');
data = readtable(filename, 'Sheet', 'Sheet1');

% 定义变量名称
data.Properties.VariableNames = {
    'ID', 'PregnantCode', 'TestTimes', 'GestationalWeeks', 'BMI', 'RawReads', ...
    'MappedRatio', 'DuplicateRatio', 'UniqueReads', 'GCContent', 'Z13', 'Z18', ...
    'Z21', 'ZX', 'ZY', 'YConcentration', 'XConcentration', 'GC13', 'GC18', ...
    'GC21', 'FilteredRatio', 'PregnancyTimes', 'ProductionTimes'};

vars = {'TestTimes', 'GestationalWeeks', 'BMI', 'RawReads', 'MappedRatio', ...
        'DuplicateRatio', 'UniqueReads', 'GCContent', 'Z13', 'Z18', 'Z21', ...
        'ZX', 'ZY', 'YConcentration', 'XConcentration', 'GC13', 'GC18', 'GC21', ...
        'FilteredRatio', 'PregnancyTimes', 'ProductionTimes'};

selectedData = data(:, vars);
selectedData = rmmissing(selectedData);    % 删除缺失值
dataMatrix = table2array(selectedData);

% Spearman秩相关系数矩阵
[spearmanRho, spearmanPval] = corr(dataMatrix, 'Type', 'Spearman');

% 绘制热力图
figure; set(gcf, 'Color', 'w');
imagesc(spearmanRho); colorbar;
caxis([-1 1]);
title('Spearman Correlation Coefficients');
xticks(1:length(vars)); yticks(1:length(vars));
x_labels = {'X21','X1','X2','X17','X18','X19','X20','X3','X4','X5','X6','X7','X8','X12','X13','X9','X10','X11','X22','X23','X24'};
xticklabels(x_labels); yticklabels(x_labels);
xtickangle(45);

% 在格内显示相关系数
for i = 1:length(vars)
    for j = 1:length(vars)
        textColor = 'w'; if abs(spearmanRho(i,j)) <= 0.7, textColor = 'k'; end
        text(j, i, num2str(spearmanRho(i,j), '%.2f'), 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'Color', textColor);
    end
end

% t检验
tTestResults = cell(length(vars)+1, length(vars)+1);
tTestResults{1,1} = 'Variables';
for i = 1:length(vars)
    tTestResults{1, i+1} = vars{i};
    tTestResults{i+1, 1} = vars{i};
end
for i = 1:length(vars)
    for j = i+1:length(vars)
        [~, pval, ~, stats] = ttest(dataMatrix(:,i), dataMatrix(:,j));
        tTestResults{i+1, j+1} = pval;
        tTestResults{j+1, i+1} = stats.tstat;
    end
end

% 保存结果
outputPath = fullfile(cfg.results_dir, 'TTest_Results.xlsx');
writecell(tTestResults, outputPath);
fprintf('分析完成，有效样本数：%d\n', size(dataMatrix,1));
