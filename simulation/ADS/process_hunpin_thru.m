%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频器直通(Thru)数据处理
% 输入: hunpin_thru.txt (ADS混频器直接输出数据, 已包含混频结果)
% 功能: 低通滤波提取差频信号 → FFT频谱分析 → 三角形校正 → 计算时延
% 参数: LFMCW 34.4-37.6 GHz, T_m = 0.5 μs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. LFMCW 参数设置
f_start = 34.4e9;           % 起始频率 (Hz)
f_end   = 37.6e9;           % 终止频率 (Hz)
B       = f_end - f_start;  % 扫频带宽 3.2 GHz
T_m     = 0.5e-6;           % 扫频周期 500 ns
K       = B / T_m;           % 调频斜率 (Hz/s)

fprintf('===== ADS Thru 数据差频提取 =====\n');
fprintf('LFMCW参数:\n');
fprintf('  频率范围:   %.1f - %.1f GHz\n', f_start/1e9, f_end/1e9);
fprintf('  带宽 B:     %.1f GHz\n', B/1e9);
fprintf('  扫频周期:   %.1f ns\n', T_m*1e9);
fprintf('  调频斜率 K: %.3e Hz/s\n', K);

%% 2. 加载数据
fprintf('\n正在加载 hunpin_thru.txt...\n');
data = readmatrix('hunpin_thru.txt', 'FileType', 'text', 'NumHeaderLines', 1);
t_raw = data(:, 1);   % 时间 (s)
v_raw = data(:, 2);   % 电压 (V)

% 去除末尾 NaN / 空行
valid_mask = ~isnan(t_raw) & ~isnan(v_raw);
t_raw = t_raw(valid_mask);
v_raw = v_raw(valid_mask);
N_total = length(t_raw);
T_data = t_raw(end) - t_raw(1);

% 显示原始数据参数
dt_typical = median(diff(t_raw(round(N_total*0.1):round(N_total*0.2))));
fprintf('  数据点数: %d\n', N_total);
fprintf('  典型采样间隔: %.3f ps\n', dt_typical*1e12);
fprintf('  数据时长: %.2f ns\n', T_data*1e9);

%% 3. 信号预处理：均匀重采样 + 低通滤波提取差频
% --- 3.1 均匀重采样 (解决ADS变步长问题) ---
fs_dec = 4e9;   % 4 GHz 均匀采样率
N_resamp = round(T_data * fs_dec);
t_uniform = linspace(t_raw(1), t_raw(end), N_resamp).';
v_uniform = interp1(t_raw, v_raw, t_uniform, 'spline');

fprintf('\n均匀重采样: fs = %.1f GHz, N = %d\n', fs_dec/1e9, N_resamp);

% --- 3.2 低通滤波提取差频信号 ---
% 混频器输出包含和频(~70GHz)与差频(~MHz量级)
% 低通滤波截止200MHz, 完全去除和频分量, 保留差频
fc_lp = 200e6;
Wn = fc_lp / (fs_dec / 2);
[b_lp, a_lp] = butter(4, Wn);
s_if = filtfilt(b_lp, a_lp, v_uniform);

fprintf('低通滤波完成: 截止 = %.0f MHz, 归一化 Wn = %.4f\n', fc_lp/1e6, Wn);

% --- 3.3 去除直流分量 ---
s_if = s_if - mean(s_if);

%% 4. FFT 频谱分析 (加汉宁窗)
N = length(s_if);
f_s = fs_dec;
df_res = f_s / N;   % 频率分辨率

% 加汉宁窗抑制旁瓣
win = hann(N);
s_if_win = s_if .* win;

S_IF = fft(s_if_win, N);
S_IF_mag = abs(S_IF) * 2;   % 补偿窗函数幅度损失

% 正频率轴
L_half = ceil(N/2);
f_axis = (0:L_half-1).' * df_res;
mag_plot = S_IF_mag(1:L_half);

fprintf('\nFFT分析: N = %d, 频率分辨率 = %.4f MHz\n', N, df_res/1e6);

%% 5. 三角形校正算法 (精确定频)
% 物理推断：纯走线带来的差频应该在极低频 (通常小于 5 MHz)
% 因此，我们只在这个狭窄的真实物理区间内寻找峰值，避开高频互调杂散
f_max_search = 5e6; % 强制限制搜索上限为 5 MHz
search_idx = find(f_axis > 50e3 & f_axis <= f_max_search); % 跳过极低频DC泄漏

if isempty(search_idx)
    error('在搜索范围内未找到有效频谱数据');
end

[val_peak, local_idx] = max(mag_plot(search_idx));
idx_peak = search_idx(local_idx);

% 三角形插值校正
if idx_peak > 1 && idx_peak < L_half
    A_L = mag_plot(idx_peak - 1);
    A_C = mag_plot(idx_peak);
    A_R = mag_plot(idx_peak + 1);
    delta_k = (A_R - A_L) / (A_L + A_C + A_R);
    f_corr = (idx_peak - 1 + delta_k) * df_res;
else
    f_corr = f_axis(idx_peak);
    delta_k = 0;
end

f_raw = f_axis(idx_peak);   % 未校正 (FFT bin 中心)

%% 6. 计算时延
tau = f_corr / K;

fprintf('\n===== 频谱分析结果 =====\n');
fprintf('未校正差频:  %.4f MHz (bin %d)\n', f_raw/1e6, idx_peak);
fprintf('校正偏移:    %.4f 个bin\n', delta_k);
fprintf('校正后差频:  %.4f MHz\n', f_corr/1e6);
fprintf('\n===== 时延计算结果 =====\n');
fprintf('校正后差频 f_beat = %.4f MHz\n', f_corr/1e6);
fprintf('调频斜率 K       = %.3e Hz/s\n', K);
fprintf('时延 τ = f_beat/K = %.4f ns\n', tau*1e9);

%% 7. 可视化

% -------------------------------------------------------------------------
% Figure 1: 原始混频器输出 + 提取的差频信号
% -------------------------------------------------------------------------
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [50, 50, 1200, 800]);

% 上: 原始混频器输出 (显示局部)
subplot(3, 1, 1);
t_disp_ns = 100;
idx_disp_raw = t_raw*1e9 <= t_disp_ns;
plot(t_raw(idx_disp_raw)*1e9, v_raw(idx_disp_raw), 'Color', [0.3 0.3 0.8], 'LineWidth', 0.5);
xlabel('时间 (ns)'); ylabel('电压 (V)');
title('原始混频器输出信号 (前100ns)');
grid on;

% 中: 差频信号 (全时段)
subplot(3, 1, 2);
plot(t_uniform*1e9, s_if, 'b', 'LineWidth', 1.0);
xlabel('时间 (ns)'); ylabel('幅值');
title('差频信号 (低通滤波 + 去直流)');
grid on;

% 下: 差频信号局部放大
subplot(3, 1, 3);
idx_disp_beat = t_uniform*1e9 <= 300;
plot(t_uniform(idx_disp_beat)*1e9, s_if(idx_disp_beat), 'b', 'LineWidth', 1.0);
xlabel('时间 (ns)'); ylabel('幅值');
title('差频信号 (前300ns放大)');
grid on;

% -------------------------------------------------------------------------
% Figure 2: 频谱分析与三角形校正
% -------------------------------------------------------------------------
figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [150, 150, 900, 750]);

% 上: 全频谱 (限制显示 0-5 MHz)
subplot(2, 1, 1);
plot(f_axis/1e6, mag_plot, 'b', 'LineWidth', 1.2);
hold on;
xline(f_corr/1e6, 'r--', 'LineWidth', 1.5);
xlabel('频率 (MHz)'); ylabel('幅值');
title('差频信号频谱 (极低频真实物理区间 0-5 MHz)');
xlim([0, 5]);
grid on;
legend('频谱', sprintf('校正频率 %.4f MHz', f_corr/1e6), 'Location', 'best');

% 下: 峰值区域放大 (stem图 + 校正标注)
subplot(2, 1, 2);
zoom_half = 0.5e6; % 缩小放大范围
stem(f_axis/1e6, mag_plot, 'b', 'MarkerSize', 5, 'LineWidth', 1.2, 'BaseValue', 0);
hold on;
xline(f_corr/1e6, 'r--', 'LineWidth', 1.5);
text(f_corr/1e6, val_peak*0.85, ...
    sprintf('  校正差频: %.4f MHz\n  时延 τ: %.4f ns', f_corr/1e6, tau*1e9), ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 10);
xlim([max(0, (f_corr - zoom_half)/1e6), (f_corr + zoom_half)/1e6]);
ylim([0, val_peak * 1.15]);
xlabel('频率 (MHz)'); ylabel('幅值');
title('[校正分析] 差频频谱峰值区域');
grid on;

% 结果标注框
annotation('textbox', [0.15, 0.48, 0.7, 0.05], ...
    'String', sprintf('校正差频 = %.4f MHz  |  时延 τ = %.4f ns', f_corr/1e6, tau*1e9), ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'k', 'LineWidth', 1, ...
    'FontWeight', 'bold', 'FontSize', 12, 'BackgroundColor', [1 1 0.9]);

fprintf('\n===== 处理完毕 =====\n');
