%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频信号时延提取 - 终稿处理版（坚持科学基准扣除）
% 1. 严格扣除系统真实时延 (0.2470 ns)
% 2. 利用幅度阈值剔除阻带噪声
% 3. 利用数据自举因果下界与局部连续性修正抑制通带失锁伪点
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 加载数据与参数
data = readmatrix('hunpin_time_v.txt', 'FileType', 'text', 'NumHeaderLines', 1);
valid = ~isnan(data(:,1)) & ~isnan(data(:,2));
t_raw = data(valid, 1); v_raw = data(valid, 2);

T_m = t_raw(end) - t_raw(1);
f_start = 34.4e9; f_end = 37.61e9; 
K = (f_end - f_start) / T_m;          

% --- 坚持科学判断：引入系统基准时延 ---
baseline_delay = 0.2470e-9; 

%% 2. 预处理（重采样 + 低通）
fs_dec = 4e9; 
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');

[b_lp, a_lp] = butter(4, 200e6 / (fs_dec / 2));
s_if = filtfilt(b_lp, a_lp, v_dec);

s_proc = s_if(1:2:end); t_proc = t_dec(1:2:end); f_s_proc = fs_dec / 2;   

% --- 混频/中频信号离散频谱 ---
N_if = length(s_if);
f_if = (0:N_if-1) * (fs_dec / N_if);
S_if = fft(s_if, N_if);
S_if_mag = abs(S_if);
S_if_mag = S_if_mag ./ max(S_if_mag + eps);

f_if_limit = 200e6;
idx_if = find(f_if <= f_if_limit);

figure('Color', 'w', 'Position', [120, 120, 900, 420]);
stem(f_if(idx_if)/1e6, S_if_mag(idx_if), 'b', 'MarkerSize', 2);
grid on;
xlabel('频率 (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('归一化幅值 (无量纲)', 'FontSize', 12, 'FontWeight', 'bold');
title('混频信号离散频谱图 (ADS: hunpin_time_v)', 'FontSize', 13);
set(gca, 'FontName', 'SimHei', 'FontSize', 11);
xlim([0, f_if_limit/1e6]); % X 轴不留空白边距

% 自动导出为论文统一风格
export_thesis_figure(gcf, 'mix_spectrum_ads_hunpin', 14, 300);

%% 3. ESPRIT 提取与盲化清洗
obs = extract_ads_delay_observations();
baseline_delay = obs.baseline_delay;
f_valid = obs.f_fit;
tau_valid = obs.tau_fit;
amp_valid = obs.amp_fit;
quality_valid = obs.quality_fit;

fprintf('\n======================================================\n');
fprintf('  ESPRIT 散点提取与质量概览\n');
fprintf('======================================================\n');
fprintf('  原始散点数: %d\n', obs.diag.raw_count);
fprintf('  幅度门限保留点: %d\n', obs.diag.amp_count);
fprintf('  自举因果下界保留点: %d\n', obs.diag.floor_count);
fprintf('  连续性修正点数: %d\n', obs.diag.repair_count);
fprintf('  综合有效散点: %d\n', obs.diag.final_count);
fprintf('  自举因果下界: %.4f ns\n', obs.tau_floor * 1e9);
fprintf('  支持度边界: t_c / T_m in [%.3f, %.3f]\n', obs.edge_margin, 1 - obs.edge_margin);
if ~isempty(quality_valid)
    fprintf('  质量指标 Q: median = %.2f, P10 = %.2f, P90 = %.2f\n', ...
        median(quality_valid), prctile(quality_valid, 10), prctile(quality_valid, 90));
end
fprintf('======================================================\n\n');

figure('Color', 'w', 'Position', [100, 100, 900, 600]);
hold on;

has_delay_curve = false;
f_true = [];
tau_true = [];
try
    delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
    f_true = delay_data(:,1);
    tau_true = delay_data(:,2);
    has_delay_curve = true;
    plot(delay_data(:,1)/1e9, delay_data(:,2)*1e9, 'r-', 'LineWidth', 2, 'DisplayName', 'ADS 群时延参考曲线');
catch
    warning('未找到 delay.txt');
end

scatter(f_valid/1e9, tau_valid*1e9, 50, amp_valid, 'filled', ...
        'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
        'DisplayName', sprintf('ESPRIT有效特征 (系统标定 \\Delta\\tau= -%.3f ns)', baseline_delay*1e9));

cb = colorbar; ylabel(cb, '中频信号有效幅度 (RMS)', 'FontSize', 11);
grid on; set(gca, 'GridAlpha', 0.3, 'FontSize', 11);
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('去嵌入后器件群时延 \\tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title('LFMCW 提取散点与去嵌入标定结果', 'FontSize', 14);

% --- 严格限制 X 轴视野，突出核心色散区域 ---
xlim([34.4, 37.6]); 
ylim([0, 8]);
legend('Location', 'northeast', 'FontSize', 11);

%% 5. 散点质量评估与物理参数自动评估输出
if has_delay_curve && ~isempty(f_valid)
    tau_true_at_scatter = interp1(f_true, tau_true, f_valid, 'pchip');
    residuals = tau_valid - tau_true_at_scatter;
    abs_residuals = abs(residuals);
    f_valid_ghz = f_valid / 1e9;

    fprintf('======================================================\n');
    fprintf('  散点质量评估（基于原始有效散点）\n');
    fprintf('======================================================\n');
    fprintf('  MAE  : %.4f ns\n', mean(abs_residuals)*1e9);
    fprintf('  RMSE : %.4f ns\n', sqrt(mean(residuals.^2))*1e9);
    fprintf('  最大偏差 : %.4f ns\n', max(abs_residuals)*1e9);
    fprintf('  平均偏置 : %.4f ns\n', mean(residuals)*1e9);
    fprintf('  标准差   : %.4f ns\n\n', std(residuals)*1e9);

    mask_flat = (f_valid_ghz >= 36.7) & (f_valid_ghz <= 37.3);
    mask_transition = ((f_valid_ghz >= 36.5) & (f_valid_ghz < 36.7)) | ...
                      ((f_valid_ghz > 37.3) & (f_valid_ghz <= 37.5));
    mask_peak = ((f_valid_ghz >= 36.43) & (f_valid_ghz < 36.5)) | (f_valid_ghz > 37.5);
    mask_unclassified = ~mask_flat & ~mask_transition & ~mask_peak;
    if any(mask_unclassified)
        mask_transition = mask_transition | mask_unclassified;
    end

    zones = {'平坦区', '过渡区', '峰值区'};
    masks = {mask_flat, mask_transition, mask_peak};

    fprintf('======================================================\n');
    fprintf('  分区精度统计\n');
    fprintf('======================================================\n');
    fprintf('%-18s | %6s | %8s | %8s | %10s | %10s\n', ...
        '分区', '点数', 'MAE(ns)', 'RMSE(ns)', 'MaxDev(ns)', 'Bias(ns)');
    fprintf('%s\n', repmat('-', 1, 78));
    for z = 1:length(zones)
        mask_zone = masks{z};
        n_pts = sum(mask_zone);
        if n_pts == 0
            fprintf('%-18s | %6d | %8s | %8s | %10s | %10s\n', ...
                zones{z}, 0, 'N/A', 'N/A', 'N/A', 'N/A');
            continue;
        end

        res_zone = residuals(mask_zone);
        abs_res_zone = abs_residuals(mask_zone);
        fprintf('%-18s | %6d | %8.4f | %8.4f | %10.4f | %+10.4f\n', ...
            zones{z}, n_pts, mean(abs_res_zone)*1e9, sqrt(mean(res_zone.^2))*1e9, ...
            max(abs_res_zone)*1e9, mean(res_zone)*1e9);
    end
    fprintf('%s\n\n', repmat('-', 1, 78));

    figure('Color', 'w', 'Position', [150, 150, 900, 450]);

    subplot(1,2,1);
    hold on;
    colors = {[0.2 0.6 0.8], [0.9 0.6 0.1], [0.8 0.2 0.2]};
    zone_labels_short = {'平坦区', '过渡区', '峰值区'};
    for z = 1:length(masks)
        mask_zone = masks{z};
        if any(mask_zone)
            scatter(f_valid(mask_zone)/1e9, residuals(mask_zone)*1e9, 48, colors{z}, 'filled', ...
                'DisplayName', zone_labels_short{z});
        end
    end
    yline(0, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    grid on;
    xlabel('瞬时探测频率 (GHz)', 'FontSize', 12);
    ylabel('残差 \\Delta\\tau (ns)', 'FontSize', 12);
    title('(a) 逐点残差分布', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 10);

    subplot(1,2,2);
    histogram(residuals*1e9, 15, 'Normalization', 'pdf', ...
        'FaceColor', [0.3 0.5 0.7], 'EdgeAlpha', 0.3);
    hold on;
    xline(0, 'k--', 'LineWidth', 1);
    xline(mean(residuals)*1e9, 'r-', 'LineWidth', 2);
    grid on;
    xlabel('残差 \\Delta\\tau (ns)', 'FontSize', 12);
    ylabel('概率密度', 'FontSize', 12);
    title('(b) 残差直方图', 'FontSize', 13);
    legend({'残差分布', '零基线', sprintf('均值 = %.4f ns', mean(residuals)*1e9)}, ...
        'Location', 'best', 'FontSize', 10);

    sgtitle('散点质量评估：仅统计，不做修正', 'FontSize', 15, 'FontWeight', 'bold');
end

if ~isempty(f_valid)
% 以 37 GHz 为界，分别寻找左右两个色散尖峰（延迟最大值）
    f_split = 37.0e9;
    
% 寻找左峰
    mask_L = f_valid < f_split;
    [~, idx_L] = max(tau_valid(mask_L));
    f_L_subset = f_valid(mask_L);
    f_peak_L = f_L_subset(idx_L);
    
% 寻找右峰
    mask_R = f_valid >= f_split;
    [~, idx_R] = max(tau_valid(mask_R));
    f_R_subset = f_valid(mask_R);
    f_peak_R = f_R_subset(idx_R);
    
% 计算绝对通带带宽
    BW_pass = f_peak_R - f_peak_L;
    
% 寻找中心频率（两峰之间的谷底，即局部延迟最小点）
    mask_C = (f_valid > f_peak_L) & (f_valid < f_peak_R);
    [~, idx_C] = min(tau_valid(mask_C));
    f_C_subset = f_valid(mask_C);
    f_center = f_C_subset(idx_C);
    
    fprintf('\n======================================================\n');
    fprintf('  基于 ESPRIT 去嵌入提取的滤波器物理参数自动评估\n');
    fprintf('======================================================\n');
    fprintf('  左侧通带边缘 (左色散峰): %7.3f GHz\n', f_peak_L / 1e9);
    fprintf('  右侧通带边缘 (右色散峰): %7.3f GHz\n', f_peak_R / 1e9);
    fprintf('  等效绝对带宽 (BW_pass) : %7.3f GHz\n', BW_pass / 1e9);
    fprintf('  滤波器中心频率 (F_center): %7.3f GHz\n', f_center / 1e9);
    fprintf('------------------------------------------------------\n');
    
    
% 相对误差计算（与理论真值 F0=37GHz, BW=1GHz 比较）
    F0_true = 37.0e9;
    BW_true = 1.0e9;
    err_F0 = abs(f_center - F0_true) / F0_true * 100;
    err_BW = abs(BW_pass - BW_true) / BW_true * 100;
    
    fprintf('  中心频率相对误差 (F_center ): %5.2f%%\n', err_F0);
    fprintf('  绝对带宽相对误差 (BW_pass ) : %5.2f%%\n', err_BW);
    fprintf('======================================================\n\n');
else
    disp('警告: 有效特征点不足，无法进行物理参数评估。');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 本地函数：统一论文插图风格并自动导出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 300; end

height_cm = width_cm * 0.618;
out_dir = fullfile(pwd, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPosition', [0, 0, width_cm, height_cm], ...
    'PaperSize', [width_cm, height_cm]);

ax_all = findall(fig_handle, 'Type', 'axes');
for i_ax = 1:numel(ax_all)
    set(ax_all(i_ax), ...
        'FontName', 'SimHei', ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.20, ...
        'TickDir', 'out');
end

line_all = findall(fig_handle, 'Type', 'line');
for i_ln = 1:numel(line_all)
    if strcmp(get(line_all(i_ln), 'LineStyle'), 'none')
        set(line_all(i_ln), 'LineWidth', 1.0);
    else
        set(line_all(i_ln), 'LineWidth', 1.5);
    end
end

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
file_emf = fullfile(out_dir, [out_name, '.emf']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
exportgraphics(fig_handle, file_emf, 'ContentType', 'vector');
fprintf('【导出】%s\n', file_tiff);
fprintf('【导出】%s\n', file_emf);
end
