% 三层分组辅助函数（怀孕次数+年龄+BMI）
clear; clc;
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'multi_factor_data.xlsx'));
data.Properties.VariableNames = {'PregnantCode','Age','BMI_Mean','PregnancyCount'};

ageGroups = {'<=26','27-29','30-31','>=32'};
ageBounds = [-Inf 26; 27 29; 30 31; 32 Inf];
bmiGroups = {'[20,28)','[28,32)','[32,36)','[36,40)','>=40'};
bmiBounds = [20 28; 28 32; 32 36; 36 40; 40 Inf];

groupResults = table();
for pregCount = 1:3
    pregData = data(data.PregnancyCount == pregCount, :);
    for a = 1:size(ageBounds,1)
        if a==1, ageData = pregData(pregData.Age <= ageBounds(1,2), :);
        elseif a==size(ageBounds,1), ageData = pregData(pregData.Age >= ageBounds(a,1), :);
        else ageData = pregData(pregData.Age >= ageBounds(a,1) & pregData.Age <= ageBounds(a,2), :);
        end
        for b = 1:size(bmiBounds,1)
            if b==1, bmiData = ageData(ageData.BMI_Mean < bmiBounds(1,2), :);
            elseif b==size(bmiBounds,1), bmiData = ageData(ageData.BMI_Mean >= bmiBounds(b,1), :);
            else bmiData = ageData(ageData.BMI_Mean >= bmiBounds(b,1) & ageData.BMI_Mean < bmiBounds(b,2), :);
            end
            if height(bmiData) > 0
                newRow = table({sprintf('Preg=%d,Age=%s,BMI=%s',pregCount,ageGroups{a},bmiGroups{b})}, ...
                               height(bmiData), 'VariableNames',{'Group','Count'});
                groupResults = [groupResults; newRow];
            end
        end
    end
end
writetable(groupResults, fullfile(cfg.results_dir, 'ThreeLayerGrouping.xlsx'));
disp(groupResults);
