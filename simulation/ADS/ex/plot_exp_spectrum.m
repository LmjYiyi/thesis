%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 差频信号离散频谱绘制（ADS 仿真 + 实测频谱仪数据）
% 输入: hunpin_thru.txt         (ADS 仿真直通时域数据)
%       ex/Trace_0005.csv       (实测无滤波器, 直通基准)
%       ex/Trace_0006.csv       (实测有滤波器, 色散等效介质)
% 风格: 与 process_hunpin_thesis_final.m 中的离散频谱图保持一致
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

f_limit = 5e6;  % 全局横坐标截止 5 MHz

%% ======== Part A: ADS 仿真直通频谱 ========
fprintf('===== ADS 仿真直通数据处理 =====\n');

% --- A1. 加载时域数据 ---
data_ads = readmatrix(fullfile(project_root, 'hunpin_thru.txt'), ...
    'FileType', 'text', 'NumHeaderLines', 1);
valid_ads = ~isnan(data_ads(:,1)) & ~isnan(data_ads(:,2));
t_ads = data_ads(valid_ads, 1);
v_ads = data_ads(valid_ads, 2);

% --- A2. 均匀重采样 + 低通滤波提取差频 ---
T_data_ads = t_ads(end) - t_ads(1);
fs_dec = 4e9;
t_uni = linspace(t_ads(1), t_ads(end), round(T_data_ads * fs_dec)).';
v_uni = interp1(t_ads, v_ads, t_uni, 'spline');

[b_lp, a_lp] = butter(4, 200e6 / (fs_dec / 2));
s_if_ads = filtfilt(b_lp, a_lp, v_uni);
s_if_ads = s_if_ads - mean(s_if_ads);   % 去直流

% --- A3. FFT → 归一化幅值 ---
N_ads = length(s_if_ads);
f_ads = (0:N_ads-1).' * (fs_dec / N_ads);
S_ads_mag = abs(fft(s_if_ads, N_ads));
S_ads_mag = S_ads_mag ./ max(S_ads_mag + eps);

fprintf('  重采样点数: %d, 频率分辨率: %.4f MHz\n', N_ads, fs_dec/N_ads/1e6);

% --- A4. 绘图 ---
idx_ads = f_ads <= f_limit;
figure('Color', 'w', 'Position', [120, 50, 900, 420]);
stem(f_ads(idx_ads) / 1e6, S_ads_mag(idx_ads), 'b', 'MarkerSize', 2);
grid on;
xlabel('频率 (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('归一化幅值 (无量纲)', 'FontSize', 12, 'FontWeight', 'bold');
title('差频信号离散频谱图（ADS仿真，直通基准）', 'FontSize', 13);
set(gca, 'FontName', 'SimHei', 'FontSize', 11);
xlim([0, f_limit / 1e6]);
export_thesis_figure(gcf, 'mix_spectrum_ads_thru', 14, 300);

%% ======== Part B: 实测频谱仪数据 ========
fprintf('\n===== 实测频谱仪数据处理 =====\n');

% --- B1. 读取 ---
data_thru   = readmatrix(fullfile(script_dir, 'data', 'Trace_0005.csv'), 'NumHeaderLines', 45);
data_filter = readmatrix(fullfile(script_dir, 'data', 'Trace_0006.csv'), 'NumHeaderLines', 45);

f_thru   = data_thru(:, 1);    P_thru   = data_thru(:, 2);
f_filter = data_filter(:, 1);  P_filter = data_filter(:, 2);

%% 2. 频谱清洗：噪底估计 + 门限抑制
% 以 10-50 MHz 高频段作为纯噪声区域，估计噪底
noise_region_thru   = P_thru(f_thru >= 10e6);
noise_region_filter = P_filter(f_filter >= 10e6);

floor_thru   = median(noise_region_thru);    % 噪底中位数 (dBm)
floor_filter = median(noise_region_filter);

margin_dB = 10;   % 门限 = 噪底 + 10 dB

fprintf('===== 噪底估计 =====\n');
fprintf('  直通基准:  %.1f dBm  →  门限 %.1f dBm\n', floor_thru,   floor_thru   + margin_dB);
fprintf('  含滤波器:  %.1f dBm  →  门限 %.1f dBm\n', floor_filter, floor_filter + margin_dB);

% 注意：直通的 beat frequency 极低（τ~0.2ns → f_beat~12kHz），
% 落在频谱仪 0 Hz bin 内，(0,1) 峰是真实信号，不做 DC 截除。

% 低于门限的 bin 置零
P_thru_clean   = P_thru;
P_thru_clean(P_thru_clean < floor_thru + margin_dB) = -Inf;

P_filter_clean = P_filter;
P_filter_clean(P_filter_clean < floor_filter + margin_dB) = -Inf;

%% 3. 转换为归一化线性幅值
amp_thru   = 10 .^ (P_thru_clean / 20);
amp_thru(isinf(P_thru_clean)) = 0;
amp_thru   = amp_thru ./ max(amp_thru + eps);

amp_filter = 10 .^ (P_filter_clean / 20);
amp_filter(isinf(P_filter_clean)) = 0;
amp_filter = amp_filter ./ max(amp_filter + eps);

%% 4. 绘图（与 ADS 仿真离散频谱风格一致）
f_limit = 5e6;  % 横坐标截止 5 MHz

% --- Figure 1: 直通基准 ---
idx_thru = f_thru <= f_limit;

figure('Color', 'w', 'Position', [120, 120, 900, 420]);
stem(f_thru(idx_thru) / 1e6, amp_thru(idx_thru), 'b', 'MarkerSize', 2);
grid on;
xlabel('频率 (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('归一化幅值 (无量纲)', 'FontSize', 12, 'FontWeight', 'bold');
title('差频信号离散频谱图（实测，直通基准）', 'FontSize', 13);
set(gca, 'FontName', 'SimHei', 'FontSize', 11);
xlim([0, f_limit / 1e6]);

export_thesis_figure(gcf, 'mix_spectrum_exp_thru', 14, 300);

% --- Figure 2: 有滤波器 ---
idx_filter = f_filter <= f_limit;

figure('Color', 'w', 'Position', [120, 520, 900, 420]);
stem(f_filter(idx_filter) / 1e6, amp_filter(idx_filter), 'b', 'MarkerSize', 2);
grid on;
xlabel('频率 (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('归一化幅值 (无量纲)', 'FontSize', 12, 'FontWeight', 'bold');
title('差频信号离散频谱图（实测，含滤波器）', 'FontSize', 13);
set(gca, 'FontName', 'SimHei', 'FontSize', 11);
xlim([0, f_limit / 1e6]);

export_thesis_figure(gcf, 'mix_spectrum_exp_filter', 14, 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 本地函数：统一论文插图风格并自动导出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 300; end

height_cm = width_cm * 0.618;
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures_export');
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
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
fprintf('【导出】%s\n', file_tiff);
end
