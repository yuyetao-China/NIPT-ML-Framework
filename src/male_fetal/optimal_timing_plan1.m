clear; clc;
cfg = config();                                 % 加载路径配置
data = readtable(fullfile(cfg.male_dir, 'male_optimal.xlsx'));
maleData = data(data.YChromosomeConcentration > 0, :);

% 定义8个BMI分组边界（由聚类分析得到）
bmiEdges = [27.1, 29.0, 30.2, 31.4, 32.6, 33.9, 35.5, 38.2, 46.9];
groupNames = {'1', '2', '3', '4', '5', '6', '7', '8'};
maleData.BMIGroup = discretize(maleData.BMI, bmiEdges, 'categorical', groupNames);

% 初始化结果表格
resultTable = table();
resultTable.BMIGroup = categorical(groupNames)';
resultTable.BMIInterval = string(resultTable.BMIGroup);
resultTable.BestTime = zeros(8, 1);
resultTable.RiskValue = zeros(8, 1);
resultTable.NumPatients = zeros(8, 1);
resultTable.StdError = zeros(8, 1);

% 风险函数参数
theta = 0.04;          % 浓度达标阈值（4%）
lambda1 = 0.6;         % 时间风险权重
lambda2 = 0.3;         % 浓度不达标风险权重
lambda3 = 0.1;         % 检测错误风险权重
phi = @(t) max(0, t - 20);          % 检测时间风险惩罚
p_fail = @(t, b) 0.05 + 0.01*(t - 10) + 0.001*(b - 30);  % 检测错误概率模型
sigma = 0.01;          % 用于噪声模拟

% 对每个BMI分组计算最佳检测时点及风险值
for g = 1:length(groupNames)
    groupData = maleData(maleData.BMIGroup == groupNames{g}, :);
    patients = unique(groupData.PatientID);
    earliestTimes = [];
    % 找出每位孕妇首次达到浓度阈值的时间
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            earliestTimes = [earliestTimes; t_detect];
        end
    end
    if ~isempty(earliestTimes)
        t0 = median(earliestTimes);          % 组内中位时间作为最佳时点
        resultTable.BestTime(g) = t0;
    else
        resultTable.BestTime(g) = NaN;
        resultTable.RiskValue(g) = NaN;
        resultTable.NumPatients(g) = length(patients);
        continue;
    end
    % 计算该组每个患者的复合风险
    risks = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            Y_value = patientData.YChromosomeConcentration(idx);
            bmi_val = patientData.BMI(idx);
            fail_prob = p_fail(t_detect, bmi_val);
            if Y_value >= theta
                beta_val = 1;                % 达标则惩罚低
            else
                beta_val = 5;                % 不达标则惩罚高
            end
            gamma_val = fail_prob;
            time_penalty = abs(t_detect - t0);
            risk = lambda1 * time_penalty + lambda2 * beta_val + lambda3 * gamma_val;
            risks = [risks; risk];
        end
    end
    if ~isempty(risks)
        resultTable.RiskValue(g) = median(risks);
    else
        resultTable.RiskValue(g) = NaN;
    end
    resultTable.NumPatients(g) = length(patients);
end

% Bootstrap估计风险值的标准误（1000次重抽样）
numBoot = 1000;
bootstrapRisks = zeros(8, numBoot);
for g = 1:length(groupNames)
    groupData = maleData(maleData.BMIGroup == groupNames{g}, :);
    patients = unique(groupData.PatientID);
    if isempty(patients), continue; end
    % 先计算该组真实中位时间（用于bootstrap时固定t0）
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            earliestTimes = [earliestTimes; t_detect];
        end
    end
    if isempty(earliestTimes)
        continue;
    end
    t0 = median(earliestTimes);
    for b = 1:numBoot
        bootPatients = datasample(patients, length(patients), 'Replace', true);
        bootRisks = [];
        for i = 1:length(bootPatients)
            patientData = groupData(strcmp(groupData.PatientID, bootPatients{i}), :);
            idx = find(patientData.YChromosomeConcentration >= theta, 1);
            if ~isempty(idx)
                t_detect = patientData.GestationalWeeks(idx);
                Y_value = patientData.YChromosomeConcentration(idx);
                bmi_val = patientData.BMI(idx);
                fail_prob = p_fail(t_detect, bmi_val);
                if Y_value >= theta
                    beta_val = 1;
                else
                    beta_val = 5;
                end
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
resultTable.StdError = std(bootstrapRisks, 0, 2, 'omitnan');

disp('BMI分组、最佳NIPT时点及风险值:');
disp(resultTable);

% 各BMI组最佳时点及风险值图
figure;
subplot(2,1,1);
errorbar(1:8, resultTable.BestTime, resultTable.StdError, 'o-', 'LineWidth', 1.5);
xlabel('BMI Group');
ylabel('Best NIPT Time (Weeks)');
title('Best NIPT Time by BMI Group with Error Bars');
set(gca, 'XTick', 1:8, 'XTickLabel', groupNames);
grid on;
subplot(2,1,2);
bar(1:8, resultTable.RiskValue);
hold on;
errorbar(1:8, resultTable.RiskValue, resultTable.StdError, 'k.', 'LineWidth', 1.5);
xlabel('BMI Group');
ylabel('Risk Value');
title('Risk Value by BMI Group');
set(gca, 'XTick', 1:8, 'XTickLabel', groupNames);
grid on;

% 保存结果到results目录
writetable(resultTable, fullfile(cfg.results_dir, 'OptimalTiming_Plan1.xlsx'));
