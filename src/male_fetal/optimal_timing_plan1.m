% 基于BMI分组求解个体化最佳检测时点（方案一）
clear; clc;
cfg = config();                                 % 加载路径配置
data = readtable(fullfile(cfg.male_dir, 'male_optimal.xlsx'));
maleData = data(data.YChromosomeConcentration > 0, :);

% 已确定的BMI分组边界（来自bmi_clustering.m）
bmiEdges = [27.1, 29.0, 30.2, 31.4, 32.6, 33.9, 35.5, 38.2, 46.9];
groupNames = {'1', '2', '3', '4', '5', '6', '7', '8'};
maleData.BMIGroup = discretize(maleData.BMI, bmiEdges, 'categorical', groupNames);

% 风险函数参数
theta = 0.04;          % 浓度达标阈值
lambda1 = 0.6;         % 时间风险权重
lambda2 = 0.3;         % 达标风险权重
lambda3 = 0.1;         % 错误风险权重
p_fail = @(t, b) 0.05 + 0.01*(t - 10) + 0.001*(b - 30);  % 检测错误概率

resultTable = table();
resultTable.BMIGroup = categorical(groupNames)';
resultTable.BMI区间 = string(resultTable.BMIGroup);
resultTable.最佳时点_周 = zeros(8, 1);
resultTable.风险值 = zeros(8, 1);
resultTable.患者数 = zeros(8, 1);
resultTable.标准误 = zeros(8, 1);

% 为每个BMI组计算最佳时点
for g = 1:length(groupNames)
    groupData = maleData(maleData.BMIGroup == groupNames{g}, :);
    patients = unique(groupData.PatientID);
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            earliestTimes = [earliestTimes; t_detect];
        end
    end
    if ~isempty(earliestTimes)
        t0 = median(earliestTimes);
        resultTable.最佳时点_周(g) = t0;
    else
        resultTable.最佳时点_周(g) = NaN;
        resultTable.风险值(g) = NaN;
        resultTable.患者数(g) = length(patients);
        continue;
    end

    % 计算该组每个孕妇的风险值
    risks = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            Y_val = patientData.YChromosomeConcentration(idx);
            bmi_val = patientData.BMI(idx);
            fail_prob = p_fail(t_detect, bmi_val);
            if Y_val >= theta
                beta_val = 1;
            else
                beta_val = 5;
            end
            gamma_val = fail_prob;
            time_penalty = abs(t_detect - t0);
            risk = lambda1 * time_penalty + lambda2 * beta_val + lambda3 * gamma_val;
            risks = [risks; risk];
        end
    end
    if ~isempty(risks)
        resultTable.风险值(g) = median(risks);
    else
        resultTable.风险值(g) = NaN;
    end
    resultTable.患者数(g) = length(patients);
end

% Bootstrap标准误
numBoot = 1000;
bootstrapRisks = zeros(8, numBoot);
for g = 1:length(groupNames)
    groupData = maleData(maleData.BMIGroup == groupNames{g}, :);
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
    t0 = median(earliestTimes);
    for b = 1:numBoot
        bootPatients = datasample(patients, length(patients), 'Replace', true);
        bootRisks = [];
        for i = 1:length(bootPatients)
            patientData = groupData(strcmp(groupData.PatientID, bootPatients{i}), :);
            idx = find(patientData.YChromosomeConcentration >= theta, 1);
            if ~isempty(idx)
                t_detect = patientData.GestationalWeeks(idx);
                Y_val = patientData.YChromosomeConcentration(idx);
                bmi_val = patientData.BMI(idx);
                fail_prob = p_fail(t_detect, bmi_val);
                if Y_val >= theta, beta_val = 1; else beta_val = 5; end
                gamma_val = fail_prob;
                time_penalty = abs(t_detect - t0);
                risk = lambda1 * time_penalty + lambda2 * beta_val + lambda3 * gamma_val;
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
resultTable.标准误 = std(bootstrapRisks, 0, 2, 'omitnan');

disp('BMI分组、最佳NIPT时点及风险值:');
disp(resultTable);

% 绘图
figure;
subplot(2,1,1);
errorbar(1:8, resultTable.最佳时点_周, resultTable.标准误, 'o-', 'LineWidth', 1.5);
xlabel('BMI组');
ylabel('最佳NIPT时点 (周)');
title('各BMI组最佳NIPT时点（含误差条）');
set(gca, 'XTick', 1:8, 'XTickLabel', groupNames);
grid on;
subplot(2,1,2);
bar(1:8, resultTable.风险值);
hold on;
errorbar(1:8, resultTable.风险值, resultTable.标准误, 'k.', 'LineWidth', 1.5);
xlabel('BMI组');
ylabel('风险值');
title('各BMI组风险值');
set(gca, 'XTick', 1:8, 'XTickLabel', groupNames);
grid on;

% 保存结果到results目录
writetable(resultTable, fullfile(cfg.results_dir, 'OptimalTiming_Plan1.xlsx'));
