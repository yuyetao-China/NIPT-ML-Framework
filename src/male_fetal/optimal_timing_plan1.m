% 基于BMI分组求解个体化最佳检测时点
clear; clc;
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'male_optimal.xlsx'));
maleData = data(data.YChromosomeConcentration > 0, :);

% 已确定的BMI分组边界
bmiEdges = [27.1, 29.0, 30.2, 31.4, 32.6, 33.9, 35.5, 38.2, 46.9];
groupNames = {'1','2','3','4','5','6','7','8'};
maleData.BMIGroup = discretize(maleData.BMI, bmiEdges, 'categorical', groupNames);

% 风险函数参数
theta = 0.04;          % 浓度达标阈值
lambda = [0.6, 0.3, 0.1];  % 时间风险、达标风险、错误风险权重
p_fail = @(t,b) 0.05 + 0.01*(t-10) + 0.001*(b-30);  % 检测错误概率

resultTable = table();
resultTable.BMIGroup = categorical(groupNames)';
resultTable.BestTime = zeros(8,1);
resultTable.RiskValue = zeros(8,1);

for g = 1:length(groupNames)
    groupData = maleData(maleData.BMIGroup == groupNames{g}, :);
    patients = unique(groupData.PatientID);
    % 寻找每个孕妇首次浓度≥4%的孕周
    earliestTimes = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            earliestTimes = [earliestTimes; patientData.GestationalWeeks(idx)];
        end
    end
    if isempty(earliestTimes), continue; end
    t0 = median(earliestTimes);          % 组内中位数为推荐时点
    resultTable.BestTime(g) = t0;
    
    % 计算该组每个孕妇的风险值
    risks = [];
    for i = 1:length(patients)
        patientData = groupData(strcmp(groupData.PatientID, patients{i}), :);
        idx = find(patientData.YChromosomeConcentration >= theta, 1);
        if ~isempty(idx)
            t_detect = patientData.GestationalWeeks(idx);
            Y_val = patientData.YChromosomeConcentration(idx);
            bmi_val = patientData.BMI(idx);
            beta_val = 1; if Y_val < theta, beta_val = 5; end
            gamma_val = p_fail(t_detect, bmi_val);
            time_penalty = abs(t_detect - t0);
            risk = lambda(1)*time_penalty + lambda(2)*beta_val + lambda(3)*gamma_val;
            risks = [risks; risk];
        end
    end
    resultTable.RiskValue(g) = median(risks);
    resultTable.NumPatients(g) = length(patients);
end

disp(resultTable);
writetable(resultTable, fullfile(cfg.results_dir, 'OptimalTiming_Plan1.xlsx'));
