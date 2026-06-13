% 混合效应模型拟合及噪声敏感性分析
clear; clc;
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'male_with_group.xlsx'), 'Sheet', 'Sheet1');
maleData = data(data.YChromosomeConcentration > 0, :);
groups = unique(maleData.Group); groups = groups(~isnan(groups));
numGroups = length(groups);

% 风险函数参数（与plan1相同）
theta = 0.04; lambda = [0.6,0.3,0.1];
p_fail = @(t,b) 0.05 + 0.01*(t-10) + 0.001*(b-30);

resultTable = table();
resultTable.Group = groups;
resultTable.BestTime = zeros(numGroups,1);
resultTable.RiskValue = zeros(numGroups,1);

% 为每组计算最佳时点和风险值
for g = 1:numGroups
    groupData = maleData(maleData.Group == groups(g), :);
    patients = unique(groupData.PatientID);
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx), earliestTimes = [earliestTimes; patientData.GestationalWeeks(idx)]; end
    end
    if isempty(earliestTimes), continue; end
    t0 = median(earliestTimes);
    resultTable.BestTime(g) = t0;
    risks = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            beta_val = 1; if patientData.YChromosomeConcentration(idx) < theta, beta_val=5; end
            gamma_val = p_fail(t_detect, patientData.BMI(idx));
            risk = lambda(1)*abs(t_detect-t0) + lambda(2)*beta_val + lambda(3)*gamma_val;
            risks = [risks; risk];
        end
    end
    resultTable.RiskValue(g) = median(risks);
end

% Bootstrap估计标准误
numBoot = 1000;
bootstrapRisks = zeros(numGroups, numBoot);
for g = 1:numGroups
    groupData = maleData(maleData.Group == groups(g), :);
    patients = unique(groupData.PatientID);
    if isempty(patients), continue; end
    % 第一次计算t0（用于bootstrap）
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx), earliestTimes = [earliestTimes; patientData.GestationalWeeks(idx)]; end
    end
    if isempty(earliestTimes), continue; end
    t0_true = median(earliestTimes);
    for b = 1:numBoot
        bootPatients = datasample(patients, length(patients), 'Replace', true);
        bootRisks = [];
        for i = 1:length(bootPatients)
            patientData = groupData(strcmp(groupData.PatientID, bootPatients{i}), :);
            idx = find(patientData.YChromosomeConcentration >= theta, 1);
            if ~isempty(idx)
                t_detect = patientData.GestationalWeeks(idx);
                beta_val = 1; if patientData.YChromosomeConcentration(idx) < theta, beta_val=5; end
                gamma_val = p_fail(t_detect, patientData.BMI(idx));
                risk = lambda(1)*abs(t_detect-t0_true) + lambda(2)*beta_val + lambda(3)*gamma_val;
                bootRisks = [bootRisks; risk];
            end
        end
        bootstrapRisks(g,b) = median(bootRisks);
    end
end
resultTable.StdError_Risk = std(bootstrapRisks,0,2,'omitnan');

% 噪声敏感性分析：向Y染色体浓度添加高斯噪声
noise_levels = [0.01,0.02,0.03,0.04,0.05];
n_reps = 100;
pass_rates = zeros(numGroups, length(noise_levels));
for g = 1:numGroups
    groupData = maleData(maleData.Group == groups(g), :);
    patients = unique(groupData.PatientID);
    for n_idx = 1:length(noise_levels)
        sigma = noise_levels(n_idx);
        temp_rates = zeros(n_reps,1);
        for rep = 1:n_reps
            pass = 0;
            for i = 1:length(patients)
                patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
                noisy = patientData.YChromosomeConcentration + sigma*randn(size(patientData.YChromosomeConcentration));
                if any(noisy >= theta), pass = pass+1; end
            end
            temp_rates(rep) = pass / length(patients);
        end
        pass_rates(g,n_idx) = mean(temp_rates);
    end
end
% 绘图和保存结果
figure; errorbar(noise_levels, pass_rates', std(pass_rates), 'o-');
xlabel('噪声水平'); ylabel('达标率'); title('Y染色体浓度高斯噪声敏感性');
saveas(gcf, fullfile(cfg.results_dir, 'NoiseSensitivity.png'));
writetable(resultTable, fullfile(cfg.results_dir, 'MixedEffect_Results.xlsx'));
