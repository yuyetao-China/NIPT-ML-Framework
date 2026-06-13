% 基于CH指数对BMI进行K‑means聚类分组
function bmi_clustering()
    cfg = config();                             % 加载路径配置
    rng('default');
    data = readtable(fullfile(cfg.male_dir, 'male_bmi_data.xlsx'));
    data.Properties.VariableNames = {'PregnantCode', 'GestationWeek', 'BMI'};
    data = data(~isnan(data.BMI), :);

    [group, id] = findgroups(data.PregnantCode);
    meanBMI = splitapply(@mean, data.BMI, group);
    fprintf('数据概览:\n');
    fprintf('孕妇数量: %d\n', length(unique(data.PregnantCode)));
    fprintf('BMI测量值总数: %d\n', height(data));
    fprintf('平均BMI范围: %.2f - %.2f\n', min(meanBMI), max(meanBMI));
    fprintf('平均BMI标准差: %.2f\n\n', std(meanBMI));

    kList = 2:9;
    fprintf('开始评估不同分组数的效果...\n');
    wcss = zeros(length(kList), 1);
    bcss = zeros(length(kList), 1);
    db_index = zeros(length(kList), 1);
    sil_scores = zeros(length(kList), 1);
    vrc = zeros(length(kList), 1);

    for i = 1:length(kList)
        k = kList(i);
        fprintf('评估 k=%d...\n', k);
        [idx, centers] = kmeans(meanBMI, k, 'Replicates', 10, 'MaxIter', 1000);
        % WCSS
        for j = 1:k
            cluster_points = meanBMI(idx == j);
            wcss(i) = wcss(i) + sum((cluster_points - centers(j)).^2);
        end
        total_mean = mean(meanBMI);
        for j = 1:k
            cluster_size = sum(idx == j);
            bcss(i) = bcss(i) + cluster_size * (centers(j) - total_mean)^2;
        end
        % Davies-Bouldin指数
        if k > 1
            db_sum = 0;
            for j = 1:k
                max_ratio = 0;
                cluster_j = meanBMI(idx == j);
                s_j = sqrt(sum((cluster_j - centers(j)).^2) / length(cluster_j));
                for l = 1:k
                    if j ~= l
                        cluster_l = meanBMI(idx == l);
                        s_l = sqrt(sum((cluster_l - centers(l)).^2) / length(cluster_l));
                        d_jl = abs(centers(j) - centers(l));
                        ratio = (s_j + s_l) / d_jl;
                        if ratio > max_ratio
                            max_ratio = ratio;
                        end
                    end
                end
                db_sum = db_sum + max_ratio;
            end
            db_index(i) = db_sum / k;
        else
            db_index(i) = Inf;
        end
        % 轮廓系数
        sil_values = silhouette(meanBMI, idx);
        sil_scores(i) = mean(sil_values);
        % VRC（Calinski-Harabasz指数）
        if k > 1
            vrc(i) = (bcss(i)/(k-1)) / (wcss(i)/(length(meanBMI)-k));
        else
            vrc(i) = 0;
        end
    end

    % 归一化加权综合得分
    normalized_scores = zeros(length(kList), 5);
    normalized_scores(:, 1) = 1 - (wcss - min(wcss)) / (max(wcss) - min(wcss));
    normalized_scores(:, 2) = (bcss - min(bcss)) / (max(bcss) - min(bcss));
    normalized_scores(:, 3) = 1 - (db_index - min(db_index)) / (max(db_index) - min(db_index));
    normalized_scores(:, 4) = (sil_scores - min(sil_scores)) / (max(sil_scores) - min(sil_scores));
    normalized_scores(:, 5) = (vrc - min(vrc)) / (max(vrc) - min(vrc));
    weights = [0.3, 0.3, 0.2, 0.1, 0.1];
    combined_scores = zeros(length(kList), 1);
    for i = 1:length(kList)
        combined_scores(i) = sum(weights .* normalized_scores(i, :));
    end

    fprintf('\n评估结果:\n');
    fprintf('k\tWCSS\t\tBCSS\t\tDB指数\t\t轮廓系数\tVRC\t\t综合得分\n');
    for i = 1:length(kList)
        fprintf('%d\t%.2f\t\t%.2f\t\t%.4f\t%.4f\t%.2f\t%.4f\n', ...
            kList(i), wcss(i), bcss(i), db_index(i), sil_scores(i), vrc(i), combined_scores(i));
    end
    [max_score, best_k_idx] = max(combined_scores);
    optimalK = kList(best_k_idx);
    fprintf('\n最佳分组数: %d (综合得分: %.4f)\n', optimalK, max_score);

    % 最终聚类，计算每个区间的边界
    [idx, centers] = kmeans(meanBMI, optimalK, 'Replicates', 20, 'MaxIter', 1000);
    [centers_sorted, sort_idx] = sort(centers);
    boundaries = zeros(optimalK-1, 1);
    for i = 1:optimalK-1
        cluster_i = meanBMI(idx == sort_idx(i));
        cluster_j = meanBMI(idx == sort_idx(i+1));
        max_i = max(cluster_i);
        min_j = min(cluster_j);
        boundaries(i) = (max_i + min_j) / 2;
    end
    intervals = cell(optimalK, 1);
    minBMI = min(meanBMI);
    maxBMI = max(meanBMI);
    intervals{1} = [minBMI, boundaries(1)];
    for i = 2:optimalK-1
        intervals{i} = [boundaries(i-1), boundaries(i)];
    end
    intervals{optimalK} = [boundaries(end), maxBMI];

    % 统计各组信息
    group_counts = zeros(optimalK, 1);
    group_std = zeros(optimalK, 1);
    group_means = zeros(optimalK, 1);
    for i = 1:optimalK
        cluster_data = meanBMI(idx == sort_idx(i));
        group_counts(i) = length(cluster_data);
        group_std(i) = std(cluster_data);
        group_means(i) = mean(cluster_data);
    end
    between_distances = zeros(optimalK, optimalK);
    for i = 1:optimalK
        for j = 1:optimalK
            between_distances(i, j) = abs(group_means(i) - group_means(j));
        end
    end

    fprintf('\nBMI分组结果 (组内差异最小化, 组间差异最大化):\n');
    fprintf('组别\tBMI区间\t\t人数\t占比\t组内标准差\t组均值\n');
    for i = 1:optimalK
        low = round(intervals{i}(1), 1);
        high = round(intervals{i}(2), 1);
        fprintf('%d\t[%.1f, %.1f]\t%d\t%.1f%%\t%.2f\t\t%.2f\n', ...
            i, low, high, group_counts(i), group_counts(i)/length(meanBMI)*100, ...
            group_std(i), group_means(i));
    end
    fprintf('\n组间距离矩阵:\n');
    fprintf('\t');
    for i = 1:optimalK
        fprintf('组%d\t', i);
    end
    fprintf('\n');
    for i = 1:optimalK
        fprintf('组%d\t', i);
        for j = 1:optimalK
            fprintf('%.2f\t', between_distances(i, j));
        end
        fprintf('\n');
    end
    avg_within_std = mean(group_std);
    min_between_dist = min(between_distances(between_distances > 0));
    fprintf('\n整体评估:\n');
    fprintf('平均组内标准差: %.2f\n', avg_within_std);
    fprintf('最小组间距离: %.2f\n', min_between_dist);
    fprintf('组内差异/组间差异比率: %.4f\n', avg_within_std/min_between_dist);

    % 调用可视化函数（完全保留）
    create_visualizations(meanBMI, intervals, kList, combined_scores, optimalK, ...
        group_std, between_distances, cfg);
end

function create_visualizations(meanBMI, intervals, kList, combined_scores, optimalK, ...
        group_std, between_distances, cfg)
    figure('Position', [100, 100, 1400, 1000]);
    subplot(2,3,[1,2]);
    histogram(meanBMI, 50, 'FaceColor', [0.7, 0.7, 0.9], 'EdgeColor', 'none');
    hold on;
    colors = lines(optimalK);
    for i = 1:optimalK
        line([intervals{i}(1) intervals{i}(1)], ylim, 'Color', colors(i,:), 'LineWidth', 2, 'LineStyle', '--');
        text(intervals{i}(1), max(ylim)*0.9, sprintf('%.1f', intervals{i}(1)), ...
            'HorizontalAlignment', 'center', 'BackgroundColor', 'white');
    end
    line([intervals{end}(2) intervals{end}(2)], ylim, 'Color', colors(end,:), 'LineWidth', 2, 'LineStyle', '--');
    text(intervals{end}(2), max(ylim)*0.9, sprintf('%.1f', intervals{end}(2)), ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'white');
    title('BMI分布与分组边界', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('BMI', 'FontSize', 12);
    ylabel('频数', 'FontSize', 12);
    grid on;

    subplot(2,3,3);
    bar_handle = bar(kList, combined_scores, 'FaceColor', [0.5, 0.8, 0.9]);
    hold on;
    optimal_idx = find(kList == optimalK);
    bar_handle.FaceColor = 'flat';
    bar_handle.CData(optimal_idx,:) = [0.9, 0.5, 0.5];
    for i = 1:length(kList)
        text(kList(i), combined_scores(i) + 0.01, sprintf('%.3f', combined_scores(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    title('不同分组数的综合得分', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('分组数 k', 'FontSize', 12);
    ylabel('综合得分', 'FontSize', 12);
    grid on;
    legend('综合得分', '最优分组', 'Location', 'best');

    subplot(2,3,4);
    bar(1:optimalK, group_std, 'FaceColor', [0.9, 0.6, 0.6]);
    title('各组组内标准差', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('组别', 'FontSize', 12);
    ylabel('标准差', 'FontSize', 12);
    grid on;

    subplot(2,3,5);
    imagesc(between_distances);
    colorbar;
    title('组间距离热图', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('组别', 'FontSize', 12);
    ylabel('组别', 'FontSize', 12);
    set(gca, 'XTick', 1:optimalK);
    set(gca, 'YTick', 1:optimalK);

    subplot(2,3,6);
    avg_within_std = mean(group_std);
    min_between_dist = min(between_distances(between_distances > 0));
    bar([1, 2], [avg_within_std, min_between_dist], 'FaceColor', [0.5, 0.7, 0.9]);
    set(gca, 'XTickLabel', {'平均组内标准差', '最小组间距离'});
    title('组内vs组间差异对比', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('数值', 'FontSize', 12);
    grid on;
    text(1, avg_within_std, sprintf('%.2f', avg_within_std), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
    text(2, min_between_dist, sprintf('%.2f', min_between_dist), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold');

    % 保存图表到results目录
    saveas(gcf, fullfile(cfg.results_dir, 'bmi_grouping_optimized_analysis.png'));
    fprintf('\n图表已保存为 %s\n', fullfile(cfg.results_dir, 'bmi_grouping_optimized_analysis.png'));
end
