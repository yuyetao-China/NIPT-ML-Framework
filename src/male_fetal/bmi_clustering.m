% 基于CH指数对BMI进行K‑means聚类分组
function bmi_clustering()
cfg = config();
data = readtable(fullfile(cfg.male_dir, 'male_bmi_data.xlsx'));
data.Properties.VariableNames = {'PregnantCode', 'GestationWeek', 'BMI'};
data = data(~isnan(data.BMI), :);

% 每个孕妇的平均BMI（每人可能有多次测量）
[~,~,group] = unique(data.PregnantCode);
meanBMI = splitapply(@mean, data.BMI, group);
kList = 2:9;
% 初始化评价指标
wcss = zeros(length(kList),1); bcss = zeros(length(kList),1);
db_index = zeros(length(kList),1); sil_scores = zeros(length(kList),1);
vrc = zeros(length(kList),1);

for i = 1:length(kList)
    k = kList(i);
    [idx, centers] = kmeans(meanBMI, k, 'Replicates', 10, 'MaxIter', 1000);
    % WCSS & BCSS
    for j = 1:k
        cluster_points = meanBMI(idx==j);
        wcss(i) = wcss(i) + sum((cluster_points - centers(j)).^2);
        bcss(i) = bcss(i) + length(cluster_points) * (centers(j) - mean(meanBMI))^2;
    end
    % Davies-Bouldin指数
    if k>1
        db_sum = 0;
        for j = 1:k
            max_ratio = 0; s_j = std(meanBMI(idx==j));
            for l = 1:k
                if j~=l
                    s_l = std(meanBMI(idx==l));
                    d_jl = abs(centers(j)-centers(l));
                    ratio = (s_j + s_l)/d_jl;
                    if ratio > max_ratio, max_ratio = ratio; end
                end
            end
            db_sum = db_sum + max_ratio;
        end
        db_index(i) = db_sum/k;
    else, db_index(i)=Inf; end
    sil_scores(i) = mean(silhouette(meanBMI, idx));
    if k>1, vrc(i) = (bcss(i)/(k-1)) / (wcss(i)/(length(meanBMI)-k)); end
end

% 归一化后加权综合得分
norm_scores = zeros(length(kList),5);
norm_scores(:,1) = 1 - (wcss-min(wcss))/(max(wcss)-min(wcss));
norm_scores(:,2) = (bcss-min(bcss))/(max(bcss)-min(bcss));
norm_scores(:,3) = 1 - (db_index-min(db_index))/(max(db_index)-min(db_index));
norm_scores(:,4) = (sil_scores-min(sil_scores))/(max(sil_scores)-min(sil_scores));
norm_scores(:,5) = (vrc-min(vrc))/(max(vrc)-min(vrc));
weights = [0.3,0.3,0.2,0.1,0.1];
combined = norm_scores * weights';
[~, best_idx] = max(combined);
optimalK = kList(best_idx);
fprintf('最优分组数：%d\n', optimalK);

% 输出BMI区间边界
[idx, centers] = kmeans(meanBMI, optimalK, 'Replicates',20,'MaxIter',1000);
[centers_sorted, sort_idx] = sort(centers);
boundaries = zeros(optimalK-1,1);
for i = 1:optimalK-1
    max_i = max(meanBMI(idx==sort_idx(i)));
    min_j = min(meanBMI(idx==sort_idx(i+1)));
    boundaries(i) = (max_i+min_j)/2;
end
intervals{1} = [min(meanBMI), boundaries(1)];
for i=2:optimalK-1, intervals{i} = [boundaries(i-1), boundaries(i)]; end
intervals{optimalK} = [boundaries(end), max(meanBMI)];
disp(intervals);
end
