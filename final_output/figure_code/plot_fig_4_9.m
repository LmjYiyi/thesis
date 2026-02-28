%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot_fig_4_9.m
% 论文图 4-9：FFT与ESPRIT特征提取方法对比
% 生成日期：2026-01-26
% 对应章节：4.4.2 特征提取框架验证
%
% 图表核心表达：
% - (a) 差频信号FFT频谱对比：强色散频谱散焦 vs 弱色散尖锐峰
% - (b) 时延轨迹全景对比：ESPRIT紧贴理论曲线，FFT发散
% - (c) 局部放大：截止频率附近的精度差异
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

fprintf('================================================\n');
fprintf(' 图4-9: FFT vs ESPRIT 特征提取方法对比\n');
fprintf('================================================\n\n');

%% 1. 物理常数与参数设置
% 【必须与表4-1保持一致】
c = 2.99792458e8;           % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
e_charge = 1.602e-19;       % 电子电荷 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 扫频起始频率 (Hz)
f_end = 37.4e9;             % 扫频终止频率 (Hz)
B = f_end - f_start;        % 扫频带宽 3.2 GHz
T_m = 50e-6;                % 调制周期 50 μs
K = B/T_m;                  % 调频斜率 6.4e13 Hz/s
f_s = 80e9;                 % 仿真采样率 80 GHz

% 传播路径参数
tau_air = 4e-9;             % 空气参考信道时延 4 ns
tau_fs = 1.75e-9;           % 自由空间单程时延 1.75 ns
d = 150e-3;                 % 等离子体层厚度 150 mm

% 等离子体参数
f_c_strong = 33e9;          % 强色散截止频率 33 GHz
f_c_weak = 25e9;            % 弱色散截止频率 25 GHz (对照)
nu = 1.5e9;                 % 碰撞频率 1.5 GHz

% 计算电子密度
omega_p_strong = 2*pi*f_c_strong;
omega_p_weak = 2*pi*f_c_weak;
n_e_strong = (omega_p_strong^2 * epsilon_0 * m_e) / e_charge^2;
n_e_weak = (omega_p_weak^2 * epsilon_0 * m_e) / e_charge^2;

% 噪声设置
SNR_dB = 20;                % 射频端信噪比 20 dB

% 采样参数
t_s = 1/f_s;
N = round(T_m/t_s);
t = (0:N-1)*t_s;

% 构建FFT频率轴 (包含负频率)
f = (0:N-1)*(f_s/N);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;

fprintf('参数配置:\n');
fprintf('  扫频范围: %.1f - %.1f GHz\n', f_start/1e9, f_end/1e9);
fprintf('  调制周期: %.0f μs\n', T_m*1e6);
fprintf('  信噪比: %d dB\n', SNR_dB);
fprintf('  强色散: f_p = %.0f GHz (n_e = %.2e m^-3)\n', f_c_strong/1e9, n_e_strong);
fprintf('  弱色散: f_p = %.0f GHz (n_e = %.2e m^-3)\n', f_c_weak/1e9, n_e_weak);
fprintf('\n');

%% 2. LFMCW信号生成
f_t = f_start + K*mod(t, T_m);
phi_t = 2*pi*cumsum(f_t)*t_s;
s_tx = cos(phi_t);

fprintf('LFMCW信号生成完成\n');

%% 3. 等离子体信道仿真

% 预处理：自由空间第一段
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];
S_after_fs1 = fft(s_after_fs1);

% 防止除以零
omega_safe = omega;
omega_safe(omega_safe == 0) = 1e-10;

% --- 强色散信道 (f_p = 33 GHz) ---
epsilon_r_strong = 1 - (omega_p_strong^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_strong(omega == 0) = 1;
k_strong = (omega ./ c) .* sqrt(epsilon_r_strong);
H_strong = exp(-1i * real(k_strong) * d - abs(imag(k_strong)) * d);

S_after_plasma_strong = S_after_fs1 .* H_strong;
s_after_plasma_strong = real(ifft(S_after_plasma_strong));
s_rx_strong = [zeros(1, delay_samples_fs) s_after_plasma_strong(1:end-delay_samples_fs)];

% 添加噪声
Ps = mean(s_rx_strong.^2);
Pn = Ps / (10^(SNR_dB/10));
rng(42);  % 可复现
s_rx_strong = s_rx_strong + sqrt(Pn) * randn(size(s_rx_strong));

% 混频与低通滤波
s_mix_strong = s_tx .* real(s_rx_strong);
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_strong = filtfilt(b_lp, a_lp, s_mix_strong);

% --- 弱色散信道 (f_p = 25 GHz) ---
epsilon_r_weak = 1 - (omega_p_weak^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_weak(omega == 0) = 1;
k_weak = (omega ./ c) .* sqrt(epsilon_r_weak);
H_weak = exp(-1i * real(k_weak) * d - abs(imag(k_weak)) * d);

S_after_plasma_weak = S_after_fs1 .* H_weak;
s_after_plasma_weak = real(ifft(S_after_plasma_weak));
s_rx_weak = [zeros(1, delay_samples_fs) s_after_plasma_weak(1:end-delay_samples_fs)];

% 添加噪声
Ps_weak = mean(s_rx_weak.^2);
Pn_weak = Ps_weak / (10^(SNR_dB/10));
s_rx_weak = s_rx_weak + sqrt(Pn_weak) * randn(size(s_rx_weak));

% 混频与低通滤波
s_mix_weak = s_tx .* real(s_rx_weak);
s_if_weak = filtfilt(b_lp, a_lp, s_mix_weak);

fprintf('等离子体信道仿真完成\n');

%% 4. FFT频谱分析

% FFT处理 (加汉宁窗)
win = hann(N)';
S_IF_strong = fft(s_if_strong .* win, N);
S_IF_weak = fft(s_if_weak .* win, N);

S_IF_strong_mag = abs(S_IF_strong) * 2;  % 补偿窗函数损失
S_IF_weak_mag = abs(S_IF_weak) * 2;

% 正频率轴
L_half = ceil(N/2);
f_axis_plot = (0:L_half-1) * (f_s / N);
mag_strong_plot = S_IF_strong_mag(1:L_half);
mag_weak_plot = S_IF_weak_mag(1:L_half);

% 峰值检测
[peak_strong, idx_peak_s] = max(mag_strong_plot);
f_peak_strong = f_axis_plot(idx_peak_s);

[peak_weak, idx_peak_w] = max(mag_weak_plot);
f_peak_weak = f_axis_plot(idx_peak_w);

% 计算3dB带宽
half_max_s = peak_strong / sqrt(2);
idx_above_s = find(mag_strong_plot > half_max_s);
bw_strong = (max(idx_above_s) - min(idx_above_s)) * (f_s / N);

half_max_w = peak_weak / sqrt(2);
idx_above_w = find(mag_weak_plot > half_max_w);
bw_weak = (max(idx_above_w) - min(idx_above_w)) * (f_s / N);

fprintf('FFT分析:\n');
fprintf('  强色散: 3dB带宽 = %.1f kHz\n', bw_strong/1e3);
fprintf('  弱色散: 3dB带宽 = %.1f kHz\n', bw_weak/1e3);
fprintf('  带宽展宽比: %.1f 倍\n', bw_strong/bw_weak);

%% 5. ESPRIT特征提取

fprintf('\n开始ESPRIT特征提取...\n');

% 数据降采样
decimation_factor = 200;
f_s_proc = f_s / decimation_factor;
s_proc = s_if_strong(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

% ESPRIT参数 (对应表4-1)
win_time = 12e-6;                % 滑动窗口 12 μs
win_len = round(win_time * f_s_proc);
step_len = round(win_len / 10);  % 90%重叠
L_sub = round(win_len / 2);

feature_f_probe = [];
feature_tau_absolute = [];
feature_amplitude = [];

num_windows = floor((N_proc - win_len) / step_len) + 1;
hWait = waitbar(0, 'ESPRIT特征提取中...');

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end
    
    x_window = s_proc(idx_start:idx_end);
    
    % 时间-频率映射
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;
    
    % 避开扫频边缘
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % Hankel矩阵构建
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end
    
    % 前后向平滑
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;
    
    % 特征值分解
    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);
    
    % MDL准则
    p = length(lambda);
    N_snaps = M_sub;
    mdl_cost = zeros(p, 1);
    for k = 0:p-1
        noise_evals = lambda(k+1:end);
        noise_evals(noise_evals < 1e-15) = 1e-15;
        g_mean = prod(noise_evals)^(1/length(noise_evals));
        a_mean = mean(noise_evals);
        term1 = -(p-k) * N_snaps * log(g_mean / a_mean);
        term2 = 0.5 * k * (2*p - k) * log(N_snaps);
        mdl_cost(k+1) = term1 + term2;
    end
    [~, min_idx] = min(mdl_cost);
    k_est = min_idx - 1;
    
    num_sources = max(1, k_est);
    num_sources = min(num_sources, 3);
    
    % TLS-ESPRIT
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    % 频率筛选
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];
    
    if mod(i, 50) == 0, waitbar(i/num_windows, hWait); end
end
close(hWait);

fprintf('ESPRIT完成: %d 个特征点\n', length(feature_f_probe));

%% 6. 传统FFT滑动估计 (对照组)

fprintf('计算FFT滑动时延估计...\n');

fft_win_time = 12e-6;
fft_win_len = round(fft_win_time * f_s);
fft_step_len = round(fft_win_len / 10);

fft_f_probe = [];
fft_tau_estimate = [];

num_fft_windows = floor((N - fft_win_len) / fft_step_len) + 1;

for i = 1:num_fft_windows
    idx_start = (i-1)*fft_step_len + 1;
    idx_end = idx_start + fft_win_len - 1;
    if idx_end > N, break; end
    
    x_window = s_if_strong(idx_start:idx_end);
    t_center = t(idx_start + round(fft_win_len/2));
    f_current_probe = f_start + K * t_center;
    
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % FFT处理
    win_fft = hann(fft_win_len)';
    X_fft = fft(x_window .* win_fft);
    mag_fft = abs(X_fft);
    
    N_fft = length(X_fft);
    f_axis_fft = (0:N_fft-1) * (f_s / N_fft);
    L_half_fft = ceil(N_fft/2);
    
    search_range = (f_axis_fft > 100e3) & (f_axis_fft < 5e6);
    mag_search = mag_fft .* search_range;
    
    [~, idx_peak] = max(mag_search(1:L_half_fft));
    
    if idx_peak > 1 && idx_peak < L_half_fft
        A_L = mag_fft(idx_peak - 1);
        A_C = mag_fft(idx_peak);
        A_R = mag_fft(idx_peak + 1);
        delta_k = (A_R - A_L) / (A_L + A_C + A_R);
        f_beat_fft = (idx_peak - 1 + delta_k) * (f_s / N_fft);
    else
        f_beat_fft = f_axis_fft(idx_peak);
    end
    
    tau_fft = f_beat_fft / K;
    
    fft_f_probe = [fft_f_probe, f_current_probe];
    fft_tau_estimate = [fft_tau_estimate, tau_fft];
end

fprintf('FFT滑动估计完成: %d 个点\n', length(fft_f_probe));

%% 7. 计算理论时延曲线

f_theory = linspace(f_start, f_end, 500);
tau_theory_strong = calculate_drude_delay(f_theory, omega_p_strong, nu, d, c);

% 相对时延
tau_esprit_relative = feature_tau_absolute - tau_air;
fft_tau_relative = fft_tau_estimate - tau_air;

% 筛选有效数据
valid_esprit = tau_esprit_relative > 0 & tau_esprit_relative < 10e-9;
tau_esprit_valid = tau_esprit_relative(valid_esprit);
f_esprit_valid = feature_f_probe(valid_esprit);
amp_esprit_valid = feature_amplitude(valid_esprit);

valid_fft = fft_tau_relative > 0 & fft_tau_relative < 10e-9;
tau_fft_valid = fft_tau_relative(valid_fft);
f_fft_valid = fft_f_probe(valid_fft);

%% 8. 计算误差统计

% ESPRIT误差
tau_theory_interp_esprit = interp1(f_theory, tau_theory_strong, f_esprit_valid, 'linear', 'extrap');
esprit_error = tau_esprit_valid - tau_theory_interp_esprit;
esprit_rmse = sqrt(mean(esprit_error.^2)) * 1e9;

% FFT误差
tau_theory_interp_fft = interp1(f_theory, tau_theory_strong, f_fft_valid, 'linear', 'extrap');
fft_error = tau_fft_valid - tau_theory_interp_fft;
fft_rmse = sqrt(mean(fft_error.^2)) * 1e9;

fprintf('\n===== 性能对比 =====\n');
fprintf('ESPRIT RMSE: %.3f ns\n', esprit_rmse);
fprintf('FFT RMSE: %.3f ns\n', fft_rmse);
fprintf('精度提升: %.1f 倍\n', fft_rmse/esprit_rmse);

%% 9. 高质量学术绘图

% 论文统一配色
colors = struct();
colors.blue = [0.0000, 0.4470, 0.7410];
colors.red = [0.8500, 0.3250, 0.0980];
colors.gray = [0.5, 0.5, 0.5];
colors.green = [0.4660, 0.6740, 0.1880];

figure('Position', [50, 50, 900, 600], 'Color', 'w');

%% ========== 时延轨迹对比（全景） ==========

% 先设置坐标轴
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

% 理论曲线
plot(f_theory/1e9, tau_theory_strong*1e9, '-', 'Color', colors.red, 'LineWidth', 2.5);

% FFT估计结果
plot(f_fft_valid/1e9, tau_fft_valid*1e9, '--', 'Color', colors.gray, 'LineWidth', 1.5);

% ESPRIT估计结果 (带幅度权重的散点)
scatter(f_esprit_valid/1e9, tau_esprit_valid*1e9, 25, amp_esprit_valid, 'filled');
colormap(gca, 'cool');
cb = colorbar;

% 中文标签
xlabel('探测频率 (GHz)', 'FontName', 'SimHei', 'FontSize', 11);
ylabel('相对群时延 Δτ (ns)', 'FontName', 'SimHei', 'FontSize', 11);
title(sprintf('时延轨迹对比 (f_p = %d GHz, SNR = %d dB)', f_c_strong/1e9, SNR_dB), ...
    'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(cb, '信号幅度权重', 'FontName', 'SimHei', 'FontSize', 10);

legend({'Drude理论曲线', '传统FFT方法', '本文ESPRIT方法'}, ...
    'Location', 'northeast', 'FontName', 'SimHei', 'FontSize', 10);

xlim([f_start/1e9, f_end/1e9]);
y_max = max(tau_theory_strong*1e9) * 1.2;
ylim([0, y_max]);
grid on;

% 高亮强色散区域
x_highlight = [34.2, 35.0];
y_lim = ylim;
patch([x_highlight(1), x_highlight(2), x_highlight(2), x_highlight(1)], ...
    [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], 'y', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
text(34.6, y_max*0.95, '强色散区', 'FontSize', 10, 'FontWeight', 'bold', ...
    'Color', [0.6 0.4 0], 'FontName', 'SimHei', 'HorizontalAlignment', 'center');

% 误差统计标注框 (移至图右下方)
annotation('textbox', [0.58, 0.18, 0.25, 0.12], ...
    'String', {sprintf('ESPRIT RMSE = %.2f ns', esprit_rmse), ...
               sprintf('FFT RMSE = %.2f ns', fft_rmse), ...
               sprintf('精度提升: %.1f 倍', fft_rmse/esprit_rmse)}, ...
    'HorizontalAlignment', 'left', 'EdgeColor', 'k', 'LineWidth', 1, ...
    'FontSize', 10, 'BackgroundColor', [1 1 0.9], 'FitBoxToText', 'on', ...
    'FontName', 'SimHei');

%% 10. 保存图表

output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存多格式
export_thesis_figure(gcf, '图4-9_FFT与ESPRIT对比', 14, 300, 'SimHei');

fprintf('\n✓ 图4-9 已保存至 final_output/figures/\n');

%% 11. 输出论文表格数据

fprintf('\n================================================\n');
fprintf(' 表4-2 数据汇总\n');
fprintf('================================================\n');
fprintf('| 性能指标 | 传统FFT峰值法 | 本文ESPRIT方法 | 性能提升 |\n');
fprintf('|----------|---------------|----------------|----------|\n');
fprintf('| 时延估计RMSE | %.2f ns | %.2f ns | %.1f倍 |\n', fft_rmse, esprit_rmse, fft_rmse/esprit_rmse);
fprintf('| 3dB带宽(强色散) | %.0f kHz | N/A(时频法) | - |\n', bw_strong/1e3);
fprintf('| 带宽展宽比 | %.1f倍 | - | - |\n', bw_strong/bw_weak);

fprintf('\n================================================\n');
fprintf(' 图4-9 生成完成！\n');
fprintf('================================================\n');

%% ========================================================================
%  局部函数定义
%% ========================================================================

function tau_rel = calculate_drude_delay(f_vec, omega_p, nu, d, c)
    % 计算Drude模型理论相对时延
    
    omega_vec = 2*pi*f_vec;
    
    % Drude模型复介电常数 (含碰撞频率)
    eps_r = 1 - (omega_p^2) ./ (omega_vec .* (omega_vec + 1i*nu));
    
    % 复波数
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 相位
    phi_plasma = -real(k_vec) * d;
    
    % 数值微分求群时延
    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    tau_total = -d_phi ./ d_omega;
    tau_total = [tau_total, tau_total(end)];
    
    % 相对时延
    tau_rel = tau_total - (d/c);
end
