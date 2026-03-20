%% plot_fig_4_6_feature_trajectory_thesis.m
% 论文图 4-6：特征轨迹重构对比（博士论文终稿版）
% 说明：
% 1) 图题不放在图内，由论文正文题注给出
% 2) 输出 TIFF(600 dpi) + PDF(矢量) + EMF(若平台支持)
% 3) 保留 colorbar，因为其承载幅度权重信息

clear; clc; close all;
rng(20260125);
fprintf('===== 图4-6: 特征轨迹重构对比（论文终稿版） =====\n');

%% 0. 绘图/导出参数
cn_font   = 'SimSun';              % 中文字体；如不稳定可改 'Microsoft YaHei'
en_font   = 'Times New Roman';     % 英文字体
font_ax   = 10.5;                  % 坐标轴刻度字号
font_lab  = 11;                    % 坐标轴标签字号
font_leg  = 10;                    % 图例字号
font_cb   = 10;                    % 色条字号
font_note = 10;                    % 截止频率标注字号

fig_width_cm  = 14.5;              % 单图宽度
fig_height_cm = 9.2;               % 单图高度
dpi_out       = 600;

%% 1. 物理常数与系统参数（与 LM_MCMC.m 保持一致）
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
e = 1.602e-19;              % 电子电量 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 起始频率 (Hz)
f_end   = 37.4e9;           % 终止频率 (Hz)
B = f_end - f_start;        % 带宽 (Hz)
T_m = 50e-6;                % 扫频周期 (s)
K = B / T_m;                % 调频斜率 (Hz/s)
f_s = 80e9;                 % 采样率 (Hz)

% 等离子体参数（强色散工况）
f_c = 33e9;                 % 截止频率 (Hz)
omega_p = 2*pi*f_c;         % 等离子体角频率
n_e = (omega_p^2 * epsilon_0 * m_e) / e^2;  %#ok<NASGU>
nu = 1.5e9;                 % 碰撞频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 自由空间传播时延
tau_fs = 1.75e-9;           % 等离子体前后自由空间时延

fprintf('系统参数: f_p = %.1f GHz, nu = %.1f GHz, d = %.2f m\n', f_c/1e9, nu/1e9, d);
fprintf('雷达参数: B = %.1f GHz, T_m = %.0f us, K = %.2e Hz/s\n', B/1e9, T_m*1e6, K);

%% 2. LFMCW 信号生成
t_s = 1/f_s;
N = round(T_m/t_s);
t = (0:N-1)*t_s;

% 构建 FFT 频率轴（含负频率）
f = (0:N-1)*(f_s/N);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;

% 生成 LFMCW 发射信号
f_t = f_start + K*mod(t, T_m);
phi_t = 2*pi*cumsum(f_t)*t_s;
s_tx = cos(phi_t);

fprintf('LFMCW 信号生成完成: N = %d 采样点\n', N);

%% 3. 等离子体传播模拟（Drude模型，频域法）
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs), s_tx(1:end-delay_samples_fs)];

S_after_fs1 = fft(s_after_fs1);

omega_safe = omega;
omega_safe(omega_safe == 0) = 1e-10;

epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_complex(omega == 0) = 1;

k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
k_real = real(k_complex);
k_imag = imag(k_complex);

H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));

s_rx_plasma = [zeros(1, delay_samples_fs), s_after_plasma(1:end-delay_samples_fs)];

fprintf('等离子体传播模拟完成\n');

%% 4. 混频与差频信号提取
s_mix_plasma = s_tx .* real(s_rx_plasma);

fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('混频处理完成\n');

%% 5. 滑动窗口 + MDL + ESPRIT 特征提取
fprintf('开始 ESPRIT 特征提取...\n');

decimation_factor = 200;
f_s_proc = f_s / decimation_factor;
s_proc = s_if_plasma(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

win_time = 12e-6;               % 窗口时长 12 us
win_len  = round(win_time * f_s_proc);
step_len = round(win_len / 10); % 90% 重叠
L_sub    = round(win_len / 2);  % 子空间维度

feature_f_probe = [];
feature_tau = [];
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

    % Hankel矩阵
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k_h = 1:M_sub
        X_hankel(:, k_h) = x_window(k_h : k_h+L_sub-1).';
    end

    % 前后向平均协方差
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;

    % 特征值分解
    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);

    % MDL
    p = length(lambda);
    N_snaps = M_sub;
    mdl_cost = zeros(p, 1);
    for k_mdl = 0:p-1
        noise_evals = lambda(k_mdl+1:end);
        noise_evals(noise_evals < 1e-15) = 1e-15;
        g_mean = prod(noise_evals)^(1/length(noise_evals));
        a_mean = mean(noise_evals);
        term1 = -(p-k_mdl) * N_snaps * log(g_mean / a_mean);
        term2 = 0.5 * k_mdl * (2*p - k_mdl) * log(N_snaps);
        mdl_cost(k_mdl+1) = term1 + term2;
    end
    [~, min_idx] = min(mdl_cost);
    k_est = min_idx - 1;

    % 物理约束
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
    if isempty(valid_freqs)
        continue;
    end

    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;

    feature_f_probe(end+1) = f_current_probe; %#ok<SAGROW>
    feature_tau(end+1) = tau_est;             %#ok<SAGROW>
    feature_amplitude(end+1) = amp_est;       %#ok<SAGROW>
end

fprintf('ESPRIT 特征提取完成: %d 个有效窗口\n', length(feature_f_probe));

%% 6. FFT 参考结果（用于对比）
fft_f_probe = [];
fft_tau = [];
fft_window_len = round(T_m * f_s / 5);  % 较大窗长

for i_fft = 1:5
    t_center_fft = (i_fft - 0.5) / 5 * T_m;
    idx_center = round(t_center_fft * f_s);
    idx_start = max(1, round(idx_center - fft_window_len/2));
    idx_end = min(N, round(idx_center + fft_window_len/2));

    x_fft_win = s_if_plasma(idx_start:idx_end) .* hann(idx_end-idx_start+1)';
    N_fft = length(x_fft_win);
    S_fft = abs(fft(x_fft_win, N_fft));

    f_fft_ax = (0:N_fft-1)*(f_s/N_fft);
    search_idx = find(f_fft_ax > 50e3 & f_fft_ax < 5e6);
    [~, peak_idx] = max(S_fft(search_idx));
    f_beat_fft = f_fft_ax(search_idx(peak_idx));

    fft_f_probe(end+1) = f_start + K * t_center_fft; %#ok<SAGROW>
    fft_tau(end+1) = f_beat_fft / K;                 %#ok<SAGROW>
end

fprintf('FFT 参考处理完成\n');

%% 7. 理论 Drude 曲线
f_theory = linspace(f_start, f_end, 500);
omega_theory = 2*pi * f_theory;

tau_theory = zeros(size(f_theory));
for i_th = 2:length(f_theory)
    omega_curr = omega_theory(i_th);
    omega_prev = omega_theory(i_th-1);

    eps_curr = 1 - (omega_p^2) / (omega_curr * (omega_curr + 1i*nu));
    eps_prev = 1 - (omega_p^2) / (omega_prev * (omega_prev + 1i*nu));

    k_curr = (omega_curr / c) * sqrt(eps_curr);
    k_prev = (omega_prev / c) * sqrt(eps_prev);

    phase_curr = -real(k_curr) * d;
    phase_prev = -real(k_prev) * d;

    tau_theory(i_th) = -(phase_curr - phase_prev) / (omega_curr - omega_prev);
end
tau_theory(1) = tau_theory(2);

tau_total_theory = tau_theory + 2 * tau_fs;
valid_theory = f_theory > f_c * 1.01;

fprintf('理论曲线计算完成\n');

%% 8. 论文终稿绘图
fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [2, 2, fig_width_cm, fig_height_cm], ...
             'PaperUnits', 'centimeters', ...
             'PaperPositionMode', 'auto', ...
             'PaperSize', [fig_width_cm, fig_height_cm]);

ax = axes(fig);
hold(ax, 'on');

% 配色：克制、论文风
clr_theory = [0.82, 0.26, 0.10];
clr_fft    = [0.50, 0.50, 0.50];

% 理论曲线
h_theory = plot(ax, ...
    f_theory(valid_theory)/1e9, ...
    tau_total_theory(valid_theory)*1e9, ...
    '-', ...
    'Color', clr_theory, ...
    'LineWidth', 1.8, ...
    'DisplayName', '理论Drude模型');

% FFT结果
h_fft = plot(ax, ...
    fft_f_probe/1e9, ...
    fft_tau*1e9, ...
    '--x', ...
    'Color', clr_fft, ...
    'LineWidth', 1.1, ...
    'MarkerSize', 7, ...
    'DisplayName', 'FFT提取');

% ESPRIT结果
amp_norm = (feature_amplitude - min(feature_amplitude)) ./ ...
           (max(feature_amplitude) - min(feature_amplitude) + eps);

h_sc = scatter(ax, ...
    feature_f_probe/1e9, ...
    feature_tau*1e9, ...
    20, ...
    amp_norm, ...
    'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', 0.3, ...
    'DisplayName', 'ESPRIT提取');

colormap(ax, winter);

% 截止频率标线
xline(ax, f_c/1e9, ':', ...
    'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.0, ...
    'HandleVisibility', 'off');

% 截止频率标注
yl = [4.5, 5.0];
text(ax, f_c/1e9 + 0.05, yl(1) + 0.10, ...
    '\fontname{Times New Roman}f_p = 33\,GHz', ...
    'Interpreter', 'tex', ...
    'FontName', en_font, ...
    'FontSize', font_note, ...
    'Rotation', 90, ...
    'Color', [0.15 0.15 0.15], ...
    'VerticalAlignment', 'bottom');

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
    'Interpreter', 'tex', 'FontSize', font_lab);
ylabel(ax, '\fontname{SimSun}总群时延 \fontname{Times New Roman}\tau_{total} (ns)', ...
    'Interpreter', 'tex', 'FontSize', font_lab);

% 坐标范围
xlim(ax, [34.0, 37.5]);
ylim(ax, yl);

% 图例
lg = legend(ax, [h_theory, h_fft, h_sc], ...
    {'理论Drude模型', 'FFT提取', 'ESPRIT提取'}, ...
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

% 色条
cb = colorbar(ax);
set(cb, ...
    'FontName', en_font, ...
    'FontSize', font_cb, ...
    'LineWidth', 0.8, ...
    'Color', [0.1 0.1 0.1]);

ylabel(cb, '\fontname{SimSun}幅度权重 \fontname{Times New Roman}A_i \fontname{SimSun}(归一化)', ...
    'Interpreter', 'tex', ...
    'FontSize', font_cb);

caxis(ax, [0, 1]);

%% 9. 导出图表
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir, 'figures_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_name_base = '图4-6_特征轨迹重构对比';

export_thesis_figure(fig, output_dir, fig_name_base, dpi_out);

fprintf('\n✓ 图 4-6 已导出\n');
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.tiff']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.pdf']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.emf']));

%% 10. 精度统计
fprintf('\n===== 精度对比统计 =====\n');

tau_theory_interp = interp1(f_theory(valid_theory), tau_total_theory(valid_theory), feature_f_probe);

valid_stat = ~isnan(tau_theory_interp) & feature_tau > 0;
error_esprit = abs(feature_tau(valid_stat) - tau_theory_interp(valid_stat));
rmse_esprit = sqrt(mean(error_esprit.^2)) * 1e9;

fprintf('ESPRIT RMSE: %.4f ns\n', rmse_esprit);
fprintf('ESPRIT 最大误差: %.4f ns\n', max(error_esprit)*1e9);
fprintf('有效特征点数: %d\n', sum(valid_stat));

tau_theory_fft = interp1(f_theory(valid_theory), tau_total_theory(valid_theory), fft_f_probe);
valid_fft_stat = ~isnan(tau_theory_fft);
error_fft = abs(fft_tau(valid_fft_stat) - tau_theory_fft(valid_fft_stat));
fprintf('FFT 平均误差: %.4f ns\n', mean(error_fft)*1e9);

%% ========================= 本地函数 =========================
function export_thesis_figure(fig_handle, out_dir, out_name, dpi)
    set(fig_handle, 'Color', 'w');

    file_tiff = fullfile(out_dir, [out_name, '.tiff']);
    file_pdf  = fullfile(out_dir, [out_name, '.pdf']);
    file_emf  = fullfile(out_dir, [out_name, '.emf']);

    exportgraphics(fig_handle, file_tiff, ...
        'Resolution', dpi, ...
        'BackgroundColor', 'white');

    exportgraphics(fig_handle, file_pdf, ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    try
        exportgraphics(fig_handle, file_emf, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    catch
        warning('EMF export failed on current platform.');
    end
end