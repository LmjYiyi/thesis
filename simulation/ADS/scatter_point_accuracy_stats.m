%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 时延特征点提取精度的定量评估
% 用途：将 ESPRIT 提取散点与 ADS 群时延真值逐点对比，输出表 5-5 所需统计量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

fprintf('======================================================\n');
fprintf('  时延特征点提取精度定量评估\n');
fprintf('  散点来源: 滑动窗口 ESPRIT + 自举清洗\n');
fprintf('  真值基准: ADS S 参数群时延 (delay.txt)\n');
fprintf('======================================================\n\n');

%% 1. Load cleaned observations shared with MCMC
obs = extract_ads_delay_observations();
X_fit = obs.f_fit;
Y_fit = obs.tau_fit;

if isempty(X_fit)
    error('有效散点为空，请检查 ESPRIT 参数或数据质量。');
end

fprintf('有效散点数: %d\n', numel(X_fit));
fprintf('清洗统计: 原始 %d -> 幅度门限 %d -> 自举下界 %d -> 最终 %d，连续性修正 %d 点\n', ...
    obs.diag.raw_count, obs.diag.amp_count, obs.diag.floor_count, ...
    obs.diag.final_count, obs.diag.repair_count);
fprintf('自举因果下界: %.4f ns\n', obs.tau_floor * 1e9);
fprintf('频率范围: %.4f - %.4f GHz\n', min(X_fit) / 1e9, max(X_fit) / 1e9);
fprintf('时延范围: %.4f - %.4f ns\n\n', min(Y_fit) * 1e9, max(Y_fit) * 1e9);

%% 2. Load ADS truth curve
delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
f_true = delay_data(:, 1);
tau_true = delay_data(:, 2);

tau_true_at_scatter = interp1(f_true, tau_true, X_fit, 'pchip');
residuals = Y_fit - tau_true_at_scatter;
abs_residuals = abs(residuals);

fprintf('======================================================\n');
fprintf('  全频段综合精度统计\n');
fprintf('======================================================\n');
fprintf('  MAE  : %.4f ns\n', mean(abs_residuals) * 1e9);
fprintf('  RMSE : %.4f ns\n', sqrt(mean(residuals.^2)) * 1e9);
fprintf('  Max  : %.4f ns\n', max(abs_residuals) * 1e9);
fprintf('  Bias : %.4f ns\n', mean(residuals) * 1e9);
fprintf('  Std  : %.4f ns\n\n', std(residuals) * 1e9);

%% 3. Region-wise statistics for Table 5-5
X_GHz = X_fit / 1e9;
mask_flat = (X_GHz >= 36.7) & (X_GHz <= 37.3);
mask_transition = ((X_GHz >= 36.5) & (X_GHz < 36.7)) | ...
                  ((X_GHz > 37.3) & (X_GHz <= 37.5));
mask_peak = ((X_GHz >= 36.43) & (X_GHz < 36.5)) | (X_GHz > 37.5);
mask_unclassified = ~mask_flat & ~mask_transition & ~mask_peak;
if any(mask_unclassified)
    mask_transition = mask_transition | mask_unclassified;
end

zones = {'通带平坦区', '色散过渡区', '双峰陡变区'};
masks = {mask_flat, mask_transition, mask_peak};

fprintf('======================================================\n');
fprintf('  分区精度统计（表 5-5）\n');
fprintf('======================================================\n');
fprintf('%-16s | %6s | %8s | %8s | %10s | %10s\n', ...
    '分区', '点数', 'MAE(ns)', 'RMSE(ns)', 'MaxDev(ns)', 'Bias(ns)');
fprintf('%s\n', repmat('-', 1, 78));
for z = 1:numel(zones)
    mask_zone = masks{z};
    n_pts = sum(mask_zone);
    if n_pts == 0
        fprintf('%-16s | %6d | %8s | %8s | %10s | %10s\n', ...
            zones{z}, 0, 'N/A', 'N/A', 'N/A', 'N/A');
        continue;
    end

    res_zone = residuals(mask_zone);
    abs_zone = abs_residuals(mask_zone);
    fprintf('%-16s | %6d | %8.4f | %8.4f | %10.4f | %+10.4f\n', ...
        zones{z}, n_pts, mean(abs_zone) * 1e9, sqrt(mean(res_zone.^2)) * 1e9, ...
        max(abs_zone) * 1e9, mean(res_zone) * 1e9);
end
fprintf('%s\n\n', repmat('-', 1, 78));

%% 4. Visualizations
figure('Color', 'w', 'Position', [100, 100, 920, 620]);
hold on;
plot(f_true / 1e9, tau_true * 1e9, 'r-', 'LineWidth', 2, ...
    'DisplayName', 'ADS 群时延真值');
scatter(X_fit / 1e9, Y_fit * 1e9, 52, obs.amp_fit, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
    'DisplayName', 'ESPRIT 清洗后散点');
cb = colorbar;
ylabel(cb, '中频信号 RMS', 'FontSize', 11);
grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12);
ylabel('群时延 \tau_g (ns)', 'FontSize', 12);
title('散点与 ADS 群时延真值逐点对比', 'FontSize', 14);
xlim([34.4, 37.6]);
ylim([0, 8]);
legend('Location', 'northwest', 'FontSize', 11);
figure_dir = fullfile(pwd, '..', '..', 'output', 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end
exportgraphics(gcf, fullfile(figure_dir, '第5章_图5-10_散点与真值对比.png'), 'Resolution', 300);

figure('Color', 'w', 'Position', [150, 150, 920, 420]);
subplot(1, 2, 1);
hold on;
zone_colors = {[0.2 0.6 0.8], [0.9 0.6 0.1], [0.8 0.2 0.2]};
for z = 1:numel(masks)
    mask_zone = masks{z};
    if any(mask_zone)
        scatter(X_fit(mask_zone) / 1e9, residuals(mask_zone) * 1e9, 48, zone_colors{z}, ...
            'filled', 'DisplayName', zones{z});
    end
end
yline(0, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12);
ylabel('残差 \Delta\tau (ns)', 'FontSize', 12);
title('(a) 逐点残差分布', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);

subplot(1, 2, 2);
histogram(residuals * 1e9, 15, 'Normalization', 'pdf', ...
    'FaceColor', [0.3 0.5 0.7], 'EdgeAlpha', 0.3);
hold on;
xline(0, 'k--', 'LineWidth', 1);
xline(mean(residuals) * 1e9, 'r-', 'LineWidth', 2);
grid on;
xlabel('残差 \Delta\tau (ns)', 'FontSize', 12);
ylabel('概率密度', 'FontSize', 12);
title('(b) 残差直方图', 'FontSize', 13);
legend({'残差分布', '零基线', sprintf('均值 = %.4f ns', mean(residuals) * 1e9)}, ...
    'Location', 'best', 'FontSize', 10);

sgtitle('时延特征点提取精度统计', 'FontSize', 15, 'FontWeight', 'bold');
exportgraphics(gcf, fullfile(figure_dir, '第5章_图5-10_残差统计.png'), 'Resolution', 300);
