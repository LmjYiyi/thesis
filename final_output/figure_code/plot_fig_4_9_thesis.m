%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot_fig_4_9b_thesis.m
% 论文图 4-9(b)：时延轨迹全景对比（博士论文终稿版）
%
% 图表表达：
% - 理论Drude曲线
% - FFT滑动估计结果
% - ESPRIT特征提取结果（颜色表示幅度权重）
%
% 说明：
% 1) 图题不放在图内，由论文正文题注给出
% 2) 输出 TIFF(600 dpi) + PDF(矢量) + EMF(若平台支持)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
rng(42);

fprintf('================================================\n');
fprintf(' 图4-9(b): 时延轨迹全景对比（论文终稿版）\n');
fprintf('================================================\n\n');

%% 0. 绘图/导出参数
cn_font   = 'SimSun';              % 中文字体；如异常可改为 'Microsoft YaHei'
en_font   = 'Times New Roman';     % 英文字体
font_ax   = 10.5;                  % 坐标轴刻度字号
font_lab  = 11;                    % 坐标轴标签字号
font_leg  = 10;                    % 图例字号
font_cb   = 10;                    % 色条字号
font_anno = 11;                    % 子图角标字号
font_note = 9.5;                   % 轻量说明字号

fig_width_cm  = 14.5;              % 单图宽度
fig_height_cm = 8.8;               % 单图高度
dpi_out       = 600;

%% 1. 物理常数与参数设置
c = 2.99792458e8;           % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
e_charge = 1.602e-19;       % 电子电荷 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 扫频起始频率 (Hz)
f_end   = 37.4e9;           % 扫频终止频率 (Hz)
B = f_end - f_start;        % 扫频带宽
T_m = 50e-6;                % 调制周期
K = B/T_m;                  % 调频斜率
f_s = 80e9;                 % 仿真采样率

% 传播路径参数
tau_air = 4e-9;             % 空气参考信道时延
tau_fs  = 1.75e-9;          % 自由空间单程时延
d = 150e-3;                 % 等离子体层厚度

% 等离子体参数（强色散工况）
f_c_strong = 33e9;          % 强色散截止频率
nu = 1.5e9;                 % 碰撞频率
omega_p_strong = 2*pi*f_c_strong;

% 噪声设置
SNR_dB = 20;

% 采样参数
t_s = 1/f_s;
N = round(T_m/t_s);
t = (0:N-1)*t_s;

% FFT频率轴（含负频率）
f = (0:N-1)*(f_s/N);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;

fprintf('参数配置:\n');
fprintf('  扫频范围: %.1f - %.1f GHz\n', f_start/1e9, f_end/1e9);
fprintf('  调制周期: %.0f μs\n', T_m*1e6);
fprintf('  信噪比: %d dB\n', SNR_dB);
fprintf('  强色散截止频率: %.0f GHz\n\n', f_c_strong/1e9);

%% 2. LFMCW信号生成
f_t = f_start + K*mod(t, T_m);
phi_t = 2*pi*cumsum(f_t)*t_s;
s_tx = cos(phi_t);

fprintf('LFMCW信号生成完成\n');

%% 3. 等离子体信道仿真（强色散）
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs), s_tx(1:end-delay_samples_fs)];
S_after_fs1 = fft(s_after_fs1);

omega_safe = omega;
omega_safe(omega_safe == 0) = 1e-10;

epsilon_r_strong = 1 - (omega_p_strong^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_strong(omega == 0) = 1;
k_strong = (omega ./ c) .* sqrt(epsilon_r_strong);
H_strong = exp(-1i * real(k_strong) * d - abs(imag(k_strong)) * d);

S_after_plasma_strong = S_after_fs1 .* H_strong;
s_after_plasma_strong = real(ifft(S_after_plasma_strong));
s_rx_strong = [zeros(1, delay_samples_fs), s_after_plasma_strong(1:end-delay_samples_fs)];

% 添加噪声
Ps = mean(s_rx_strong.^2);
Pn = Ps / (10^(SNR_dB/10));
s_rx_strong = s_rx_strong + sqrt(Pn) * randn(size(s_rx_strong));

% 混频与低通滤波
s_mix_strong = s_tx .* real(s_rx_strong);
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_strong = filtfilt(b_lp, a_lp, s_mix_strong);

fprintf('等离子体信道仿真完成\n');

%% 4. ESPRIT特征提取
fprintf('开始ESPRIT特征提取...\n');

decimation_factor = 200;
f_s_proc = f_s / decimation_factor;
s_proc = s_if_strong(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

win_time = 12e-6;
win_len = round(win_time * f_s_proc);
step_len = round(win_len / 10);
L_sub = round(win_len / 2);

feature_f_probe = [];
feature_tau_absolute = [];
feature_amplitude = [];

num_windows = floor((N_proc - win_len) / step_len) + 1;

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc
        break;
    end

    x_window = s_proc(idx_start:idx_end);

    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;

    if t_center > 0.95*T_m || t_center < 0.05*T_m
        continue;
    end

    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for kk = 1:M_sub
        X_hankel(:, kk) = x_window(kk : kk+L_sub-1).';
    end

    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;

    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);

    p = length(lambda);
    N_snaps = M_sub;
    mdl_cost = zeros(p, 1);
    for kk = 0:p-1
        noise_evals = lambda(kk+1:end);
        noise_evals(noise_evals < 1e-15) = 1e-15;
        g_mean = prod(noise_evals)^(1/length(noise_evals));
        a_mean = mean(noise_evals);
        term1 = -(p-kk) * N_snaps * log(g_mean / a_mean);
        term2 = 0.5 * kk * (2*p - kk) * log(N_snaps);
        mdl_cost(kk+1) = term1 + term2;
    end
    [~, min_idx] = min(mdl_cost);
    k_est = min_idx - 1;

    num_sources = max(1, k_est);
    num_sources = min(num_sources, 3);

    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));

    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    if isempty(valid_freqs)
        continue;
    end

    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;

    feature_f_probe(end+1) = f_current_probe;      %#ok<SAGROW>
    feature_tau_absolute(end+1) = tau_est;         %#ok<SAGROW>
    feature_amplitude(end+1) = amp_est;            %#ok<SAGROW>
end

fprintf('ESPRIT完成: %d 个特征点\n', length(feature_f_probe));

%% 5. FFT滑动估计（对照组）
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
    if idx_end > N
        break;
    end

    x_window = s_if_strong(idx_start:idx_end);
    t_center = t(idx_start + round(fft_win_len/2));
    f_current_probe = f_start + K * t_center;

    if t_center > 0.95*T_m || t_center < 0.05*T_m
        continue;
    end

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

    fft_f_probe(end+1) = f_current_probe;      %#ok<SAGROW>
    fft_tau_estimate(end+1) = tau_fft;         %#ok<SAGROW>
end

fprintf('FFT滑动估计完成: %d 个点\n', length(fft_f_probe));

%% 6. 理论时延曲线
f_theory = linspace(f_start, f_end, 500);
tau_theory_strong = calculate_drude_delay(f_theory, omega_p_strong, nu, d, c);

tau_esprit_relative = feature_tau_absolute - tau_air;
fft_tau_relative = fft_tau_estimate - tau_air;

valid_esprit = tau_esprit_relative > 0 & tau_esprit_relative < 10e-9;
tau_esprit_valid = tau_esprit_relative(valid_esprit);
f_esprit_valid = feature_f_probe(valid_esprit);
amp_esprit_valid = feature_amplitude(valid_esprit);

valid_fft = fft_tau_relative > 0 & fft_tau_relative < 10e-9;
tau_fft_valid = fft_tau_relative(valid_fft);
f_fft_valid = fft_f_probe(valid_fft);

%% 7. 误差统计
tau_theory_interp_esprit = interp1(f_theory, tau_theory_strong, f_esprit_valid, 'linear', 'extrap');
esprit_error = tau_esprit_valid - tau_theory_interp_esprit;
esprit_rmse = sqrt(mean(esprit_error.^2)) * 1e9;

tau_theory_interp_fft = interp1(f_theory, tau_theory_strong, f_fft_valid, 'linear', 'extrap');
fft_error = tau_fft_valid - tau_theory_interp_fft;
fft_rmse = sqrt(mean(fft_error.^2)) * 1e9;

fprintf('\n===== 性能对比 =====\n');
fprintf('ESPRIT RMSE: %.3f ns\n', esprit_rmse);
fprintf('FFT RMSE: %.3f ns\n', fft_rmse);
fprintf('精度提升: %.1f 倍\n', fft_rmse/esprit_rmse);

%% 8. 单图绘制（仅保留全景对比）
colors = struct();
colors.blue = [0.00, 0.32, 0.74];
colors.red  = [0.82, 0.26, 0.10];
colors.gray = [0.50, 0.50, 0.50];

fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [2, 2, fig_width_cm, fig_height_cm], ...
             'PaperUnits', 'centimeters', ...
             'PaperPositionMode', 'auto', ...
             'PaperSize', [fig_width_cm, fig_height_cm]);

ax = axes(fig);
hold(ax, 'on');

% 理论曲线
h_theory = plot(ax, f_theory/1e9, tau_theory_strong*1e9, '-', ...
    'Color', colors.red, ...
    'LineWidth', 1.8, ...
    'DisplayName', '理论曲线');

% FFT结果
h_fft = plot(ax, f_fft_valid/1e9, tau_fft_valid*1e9, '--', ...
    'Color', colors.gray, ...
    'LineWidth', 1.2, ...
    'DisplayName', 'FFT提取');

% ESPRIT结果
amp_norm = (amp_esprit_valid - min(amp_esprit_valid)) ./ ...
           (max(amp_esprit_valid) - min(amp_esprit_valid) + eps);

h_esprit = scatter(ax, ...
    f_esprit_valid/1e9, ...
    tau_esprit_valid*1e9, ...
    18, ...
    amp_norm, ...
    'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', 0.25, ...
    'DisplayName', 'ESPRIT提取');

colormap(ax, cool);
cb = colorbar(ax);
set(cb, ...
    'FontName', en_font, ...
    'FontSize', font_cb, ...
    'LineWidth', 0.8, ...
    'Color', [0.1 0.1 0.1]);

ylabel(cb, '\fontname{SimSun}幅度权重 \fontname{Times New Roman}A_i', ...
    'Interpreter', 'tex', ...
    'FontSize', font_cb);

% 轻量高亮强色散区
y_max = max(tau_theory_strong*1e9) * 1.18;
patch(ax, [34.2 35.0 35.0 34.2], [0 0 y_max y_max], [0.93 0.93 0.93], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');

uistack(h_theory, 'top');
uistack(h_fft, 'top');
uistack(h_esprit, 'top');

% 坐标轴格式
set(ax, ...
    'FontName', en_font, ...
    'FontSize', font_ax, ...
    'LineWidth', 0.9, ...
    'Box', 'on', ...
    'TickDir', 'in', ...
    'XGrid', 'on', ...
    'YGrid', 'on', ...
    'GridAlpha', 0.18, ...
    'GridLineStyle', '-');

xlabel(ax, '\fontname{SimSun}探测频率 \fontname{Times New Roman}f_{probe} (GHz)', ...
    'Interpreter', 'tex', ...
    'FontSize', font_lab);

ylabel(ax, '\fontname{SimSun}相对群时延 \fontname{Times New Roman}\Delta\tau (ns)', ...
    'Interpreter', 'tex', ...
    'FontSize', font_lab);

xlim(ax, [f_start/1e9, f_end/1e9]);
ylim(ax, [0, y_max]);

% 图例
lg = legend(ax, [h_theory, h_fft, h_esprit], ...
    {'理论曲线', 'FFT提取', 'ESPRIT提取'}, ...
    'Location', 'northeast');
set(lg, ...
    'FontName', cn_font, ...
    'FontSize', font_leg, ...
    'Interpreter', 'tex', ...
    'Box', 'on', ...
    'Color', 'white', ...
    'EdgeColor', [0.7 0.7 0.7], ...
    'LineWidth', 0.6, ...
    'AutoUpdate', 'off');



%% 9. 导出图表
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir, 'figures_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_name_base = '图4-9b_时延轨迹全景对比';
export_thesis_figure(fig, output_dir, fig_name_base, dpi_out);

fprintf('\n✓ 图4-9(b) 已导出\n');
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.tiff']));

%% 10. 输出表格数据
fprintf('\n================================================\n');
fprintf(' 数据汇总\n');
fprintf('================================================\n');
fprintf('| 性能指标 | 传统FFT峰值法 | 本文ESPRIT方法 | 性能提升 |\n');
fprintf('|----------|---------------|----------------|----------|\n');
fprintf('| 时延估计RMSE | %.2f ns | %.2f ns | %.1f倍 |\n', fft_rmse, esprit_rmse, fft_rmse/esprit_rmse);

fprintf('\n================================================\n');
fprintf(' 图4-9(b) 生成完成！\n');
fprintf('================================================\n');

%% ========================================================================
% 局部函数定义
%% ========================================================================

function export_thesis_figure(fig_handle, out_dir, out_name, dpi)
    set(fig_handle, 'Color', 'w');

    file_tiff = fullfile(out_dir, [out_name, '.tiff']);

    exportgraphics(fig_handle, file_tiff, ...
        'Resolution', dpi, ...
        'BackgroundColor', 'white');
end
function tau_rel = calculate_drude_delay(f_vec, omega_p, nu, d, c)
    omega_vec = 2*pi*f_vec;

    eps_r = 1 - (omega_p^2) ./ (omega_vec .* (omega_vec + 1i*nu));
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);

    phi_plasma = -real(k_vec) * d;

    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    tau_total = -d_phi ./ d_omega;
    tau_total = [tau_total, tau_total(end)];

    tau_rel = tau_total - (d/c);
end

