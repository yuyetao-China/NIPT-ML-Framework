% 混合效应模型拟合及噪声敏感性分析
clear; clc;
cfg = config();                                 % 加载路径配置
data = readtable(fullfile(cfg.male_dir, 'male_with_group.xlsx'), 'Sheet', 'Sheet1');
maleData = data(data.YChromosomeConcentration > 0, :);
groups = unique(maleData.Group);
groups = groups(~isnan(groups));
numGroups = length(groups);

% 风险函数参数
theta = 0.04;
lambda1 = 0.6;
lambda2 = 0.3;
lambda3 = 0.1;
p_fail = @(t, b) 0.05 + 0.01*(t - 10) + 0.001*(b - 30);

resultTable = table();
resultTable.('分组') = groups;
resultTable.('最佳时点_周') = zeros(numGroups, 1);
resultTable.('风险值') = zeros(numGroups, 1);
resultTable.('患者数') = zeros(numGroups, 1);
resultTable.('风险标准误') = zeros(numGroups, 1);
resultTable.('时点标准误') = zeros(numGroups, 1);

% 为每组计算最佳时点和风险值
for g = 1:numGroups
    groupID = groups(g);
    groupData = maleData(maleData.Group == groupID, :);
    patients = unique(groupData.PatientID);
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            earliestTimes = [earliestTimes; patientData.GestationalWeeks(idx)];
        end
    end
    if isempty(earliestTimes), continue; end
    t0 = median(earliestTimes);
    resultTable.('最佳时点_周')(g) = t0;
    risks = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            beta_val = 1; if patientData.YChromosomeConcentration(idx) < theta, beta_val=5; end
            gamma_val = p_fail(t_detect, patientData.BMI(idx));
            risk = lambda1*abs(t_detect-t0) + lambda2*beta_val + lambda3*gamma_val;
            risks = [risks; risk];
        end
    end
    if ~isempty(risks)
        resultTable.('风险值')(g) = median(risks);
    else
        resultTable.('风险值')(g) = NaN;
    end
    resultTable.('患者数')(g) = length(patients);
end

% Bootstrap估计标准误
numBoot = 1000;
bootstrapRisks = zeros(numGroups, numBoot);
bootstrapTimes = zeros(numGroups, numBoot);
for g = 1:numGroups
    groupID = groups(g);
    groupData = maleData(maleData.Group == groupID, :);
    patients = unique(groupData.PatientID);
    if isempty(patients), continue; end
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            earliestTimes = [earliestTimes; patientData.GestationalWeeks(idx)];
        end
    end
    if isempty(earliestTimes), continue; end
    t0_true = median(earliestTimes);
    for b = 1:numBoot
        bootPatients = datasample(patients, length(patients), 'Replace', true);
        bootEarliestTimes = [];
        for i = 1:length(bootPatients)
            patientData = groupData(strcmp(groupData.PatientID, bootPatients{i}), :);
            idx = find(patientData.YChromosomeConcentration >= theta, 1);
            if ~isempty(idx)
                bootEarliestTimes = [bootEarliestTimes; patientData.GestationalWeeks(idx)];
            end
        end
        if ~isempty(bootEarliestTimes)
            t0_boot = median(bootEarliestTimes);
            bootstrapTimes(g, b) = t0_boot;
        else
            bootstrapTimes(g, b) = NaN;
        end
        bootRisks = [];
        for i = 1:length(bootPatients)
            patientData = groupData(strcmp(groupData.PatientID, bootPatients{i}), :);
            idx = find(patientData.YChromosomeConcentration >= theta, 1);
            if ~isempty(idx)
                t_detect = patientData.GestationalWeeks(idx);
                beta_val = 1; if patientData.YChromosomeConcentration(idx) < theta, beta_val=5; end
                gamma_val = p_fail(t_detect, patientData.BMI(idx));
                risk = lambda1*abs(t_detect-t0_true) + lambda2*beta_val + lambda3*gamma_val;
                bootRisks = [bootRisks; risk];
            end
        end
        if ~isempty(bootRisks)
            bootstrapRisks(g, b) = median(bootRisks);
        else
            bootstrapRisks(g, b) = NaN;
        end
    end
end
resultTable.('风险标准误') = std(bootstrapRisks, 0, 2, 'omitnan');
resultTable.('时点标准误') = std(bootstrapTimes, 0, 2, 'omitnan');

disp('分组、最佳NIPT时点及风险值:');
disp(resultTable);
writetable(resultTable, fullfile(cfg.results_dir, 'MixedEffect_Results.xlsx'));

% 每个孕妇的最佳时点
patientIDs = unique(maleData.PatientID);
individualResults = table();
individualResults.PatientID = patientIDs;
individualResults.('分组') = zeros(length(patientIDs), 1);
individualResults.('最佳时点_周') = zeros(length(patientIDs), 1);
for i = 1:length(patientIDs)
    patientID = patientIDs{i};
    patientData = maleData(strcmp(maleData.PatientID, patientID), :);
    individualResults.('分组')(i) = patientData.Group(1);
    idx = find(patientData.YChromosomeConcentration >= theta, 1);
    if ~isempty(idx)
        individualResults.('最佳时点_周')(i) = patientData.GestationalWeeks(idx);
    else
        individualResults.('最佳时点_周')(i) = NaN;
    end
end
writetable(individualResults, fullfile(cfg.results_dir, 'Individual_BestTime.xlsx'));
disp(['每个孕妇的最佳时点已保存到: ' fullfile(cfg.results_dir, 'Individual_BestTime.xlsx')]);

% 绘图
figure;
subplot(2,1,1);
bar(1:numGroups, resultTable.('最佳时点_周'));
hold on;
errorbar(1:numGroups, resultTable.('最佳时点_周'), resultTable.('时点标准误'), 'k.', 'LineWidth', 1.5);
xlabel('分组');
ylabel('最佳NIPT时点 (周)');
title('各分组最佳NIPT时点（含误差条）');
set(gca, 'XTick', 1:numGroups, 'XTickLabel', groups);
grid on;
subplot(2,1,2);
bar(1:numGroups, resultTable.('风险值'));
hold on;
errorbar(1:numGroups, resultTable.('风险值'), resultTable.('风险标准误'), 'k.', 'LineWidth', 1.5);
xlabel('分组');
ylabel('风险值');
title('各分组风险值（含误差条）');
set(gca, 'XTick', 1:numGroups, 'XTickLabel', groups);
grid on;
set(gcf, 'Position', [100, 100, 800, 600]);

% 噪声敏感性分析
fprintf('开始噪声敏感性分析：Y染色体浓度添加高斯噪声...\n');
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05];
n_reps = 100;
pass_rates = zeros(numGroups, length(noise_levels));
pass_rates_std = zeros(numGroups, length(noise_levels));
for g = 1:numGroups
    groupID = groups(g);
    groupData = maleData(maleData.Group == groupID, :);
    patients = unique(groupData.PatientID);
    if isempty(patients), continue; end
    for n_idx = 1:length(noise_levels)
        sigma = noise_levels(n_idx);
        temp_rates = zeros(n_reps, 1);
        for rep = 1:n_reps
            pass = 0;
            for i = 1:length(patients)
                patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
                noisy = patientData.YChromosomeConcentration + sigma*randn(size(patientData.YChromosomeConcentration));
                if any(noisy >= theta), pass = pass+1; end
            end
            temp_rates(rep) = pass / length(patients);
        end
        pass_rates(g, n_idx) = mean(temp_rates);
        pass_rates_std(g, n_idx) = std(temp_rates);
    end
end
for g = 1:numGroups
    fprintf('组 %d:\n', groups(g));
    for n_idx = 1:length(noise_levels)
        fprintf('  噪声水平 %.3f: 达标率 = %.4f ± %.4f\n', ...
                noise_levels(n_idx), pass_rates(g, n_idx), pass_rates_std(g, n_idx));
    end
    fprintf('\n');
end
figure;
colors = lines(numGroups);
hold on;
for g = 1:numGroups
    errorbar(noise_levels, pass_rates(g, :), pass_rates_std(g, :), ...
             'o-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', colors(g, :), ...
             'DisplayName', sprintf('组 %d', groups(g)));
end
xlabel('噪声水平');
ylabel('达标率');
title('Y染色体浓度添加高斯噪声对达标率的影响');
legend('show', 'Location', 'best');
grid on;
set(gcf, 'Position', [100, 100, 900, 600]);
noiseSensResults = table();
noiseSensResults.('分组') = groups;
for n_idx = 1:length(noise_levels)
    noiseSensResults.(sprintf('噪声_%.3f_均值', noise_levels(n_idx))) = pass_rates(:, n_idx);
    noiseSensResults.(sprintf('噪声_%.3f_标准差', noise_levels(n_idx))) = pass_rates_std(:, n_idx);
end
writetable(noiseSensResults, fullfile(cfg.results_dir, 'Noise_Sensitivity_Analysis_Results.xlsx'));
disp(['噪声敏感性分析结果已保存到 ' fullfile(cfg.results_dir, 'Noise_Sensitivity_Analysis_Results.xlsx')]);
