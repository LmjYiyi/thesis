%% plot_fig_4_6_feature_trajectory.m
% 论文图 4-6：特征轨迹重构对比 (基于真实信号处理流程)
% 生成日期：2026-01-25
% 对应章节：4.2.3 基于TLS-ESPRIT的"频率-时延"特征轨迹高精度重构
%
% 图表描述（来自定稿文档）：
% - 典型工况：f_p = 33 GHz, d = 0.15 m
% - FFT（灰色虚线）：因栅栏效应呈现阶梯状跳变，强色散区完全发散
% - ESPRIT（蓝色散点）：精准描绘理论曲线，颜色深浅代表幅度权重
% - 理论Drude（红色实线）：包括纳秒级非线性弯曲
% - 重点展示：在34-35 GHz高非线性区的对比差异
%
% 信号处理流程参考：thesis-code/LM_MCMC.m

clear; clc; close all;
fprintf('===== 图4-6: 特征轨迹重构对比 (真实信号处理) =====\n');

%% 1. 物理常数与系统参数 (与LM_MCMC.m保持一致)
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
e = 1.602e-19;              % 电子电量 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 起始频率 (Hz)
f_end = 37.4e9;             % 终止频率 (Hz)
B = f_end - f_start;        % 带宽 (Hz)
T_m = 50e-6;                % 扫频周期 (s)
K = B / T_m;                % 调频斜率 (Hz/s)
f_s = 80e9;                 % 采样率 (Hz)

% 等离子体参数 (强色散工况)
f_c = 33e9;                 % 截止频率 (Hz) - 与定稿一致
omega_p = 2*pi*f_c;         % 等离子体角频率
n_e = (omega_p^2 * epsilon_0 * m_e) / e^2;  % 电子密度
nu = 1.5e9;                 % 碰撞频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 自由空间传播时延
tau_fs = 1.75e-9;           % 等离子体前后自由空间时延

fprintf('系统参数: f_p = %.1f GHz, nu = %.1f GHz, d = %.2f m\n', f_c/1e9, nu/1e9, d);
fprintf('雷达参数: B = %.1f GHz, T_m = %.0f us, K = %.2e Hz/s\n', B/1e9, T_m*1e6, K);

%% 2. LFMCW信号生成
t_s = 1/f_s;
N = round(T_m/t_s);
t = (0:N-1)*t_s;

% 构建FFT频率轴 (含负频率)
f = (0:N-1)*(f_s/N);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;

% 生成LFMCW发射信号
f_t = f_start + K*mod(t, T_m);
phi_t = 2*pi*cumsum(f_t)*t_s;
s_tx = cos(phi_t);

fprintf('LFMCW信号生成完成: N = %d 采样点\n', N);

%% 3. 等离子体传播模拟 (Drude模型, 频域法)

% 第一段: 自由空间
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs), s_tx(1:end-delay_samples_fs)];

% 第二段: 等离子体层 (频域处理)
S_after_fs1 = fft(s_after_fs1);

% 防止除零
omega_safe = omega;
omega_safe(omega_safe == 0) = 1e-10;

% Drude复介电常数
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_complex(omega == 0) = 1;

% 复波数
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
k_real = real(k_complex);
k_imag = imag(k_complex);

% 传递函数 (强制衰减)
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

% 应用传递函数
S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));

% 第三段: 自由空间
s_rx_plasma = [zeros(1, delay_samples_fs), s_after_plasma(1:end-delay_samples_fs)];

fprintf('等离子体传播模拟完成\n');

%% 4. 混频与差频信号提取
s_mix_plasma = s_tx .* real(s_rx_plasma);

% 低通滤波
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('混频处理完成\n');

%% 5. 滑动窗口 + MDL + ESPRIT 特征提取

fprintf('开始ESPRIT特征提取...\n');

% 数据预处理 (降采样)
decimation_factor = 200;
f_s_proc = f_s / decimation_factor;
s_proc = s_if_plasma(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

% 滑动窗口参数
win_time = 12e-6;               % 窗口时长 12us
win_len = round(win_time * f_s_proc);
step_len = round(win_len / 10); % 90%重叠
L_sub = round(win_len / 2);     % 子空间维度

% 存储结果
feature_f_probe = [];
feature_tau = [];
feature_amplitude = [];

% 处理循环
num_windows = floor((N_proc - win_len) / step_len) + 1;

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
    
    % 构造Hankel矩阵
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k_h = 1:M_sub
        X_hankel(:, k_h) = x_window(k_h : k_h+L_sub-1).';
    end
    
    % 前后向平均协方差估计
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
    
    % 频率筛选 (直达波: 最小频率)
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    [f_beat_est, ~] = min(valid_freqs);
    
    % 幅度提取 (用于加权)
    amp_est = rms(x_window);
    
    % 时延计算
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau = [feature_tau, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];
end

fprintf('ESPRIT特征提取完成: %d 个有效窗口\n', length(feature_f_probe));

%% 6. FFT全局处理 (用于对比)

% 全局FFT
S_IF_plasma = fft(s_if_plasma .* hann(N)', N);
S_IF_mag = abs(S_IF_plasma) * 2;

% 寻找FFT峰值 (简化: 在不同频段估计)
f_fft_axis = (0:N-1)*(f_s/N);
f_fft_valid = f_fft_axis(1:round(10e6/(f_s/N)));  % 0-10MHz
S_fft_valid = S_IF_mag(1:round(10e6/(f_s/N)));

% 模拟FFT在不同窗口的结果 (因全局FFT无法区分时频)
% 这里用短时FFT模拟
fft_f_probe = [];
fft_tau = [];
fft_window_len = round(T_m * f_s / 5);  % FFT窗口 (较大)

for i_fft = 1:5
    t_center_fft = (i_fft - 0.5) / 5 * T_m;
    idx_center = round(t_center_fft * f_s);
    idx_start = max(1, idx_center - fft_window_len/2);
    idx_end = min(N, idx_center + fft_window_len/2);
    
    % 截取窗口
    x_fft_win = s_if_plasma(idx_start:idx_end) .* hann(idx_end-idx_start+1)';
    N_fft = length(x_fft_win);
    S_fft = abs(fft(x_fft_win, N_fft));
    
    % 找峰值
    f_fft_ax = (0:N_fft-1)*(f_s/N_fft);
    search_idx = find(f_fft_ax > 50e3 & f_fft_ax < 5e6);
    [~, peak_idx] = max(S_fft(search_idx));
    f_beat_fft = f_fft_ax(search_idx(peak_idx));
    
    fft_f_probe = [fft_f_probe, f_start + K * t_center_fft];
    fft_tau = [fft_tau, f_beat_fft / K];
end

fprintf('FFT参考处理完成\n');

%% 7. 理论Drude曲线计算

f_theory = linspace(f_start, f_end, 500);
omega_theory = 2*pi * f_theory;

% 数值求导计算群时延（等离子体段）
tau_theory = zeros(size(f_theory));
for i_th = 2:length(f_theory)
    omega_curr = omega_theory(i_th);
    omega_prev = omega_theory(i_th-1);
    
    % 复介电常数
    eps_curr = 1 - (omega_p^2) / (omega_curr * (omega_curr + 1i*nu));
    eps_prev = 1 - (omega_p^2) / (omega_prev * (omega_prev + 1i*nu));
    
    % 相位
    k_curr = (omega_curr / c) * sqrt(eps_curr);
    k_prev = (omega_prev / c) * sqrt(eps_prev);
    phase_curr = -real(k_curr) * d;
    phase_prev = -real(k_prev) * d;
    
% 群时延（等离子体段）
    tau_theory(i_th) = -(phase_curr - phase_prev) / (omega_curr - omega_prev);
end
tau_theory(1) = tau_theory(2);

% 总群时延 = 等离子体段时延 + 两段自由空间时延
tau_total_theory = tau_theory + 2 * tau_fs;

% 过滤截止区
valid_theory = f_theory > f_c * 1.01;

fprintf('理论曲线计算完成\n');

%% 8. 高质量绘图

figure('Position', [100, 100, 900, 600], 'Color', 'w');

% 颜色方案
colors = struct();
colors.theory = [0.8500, 0.3250, 0.0980];  % 橙红
colors.fft = [0.5, 0.5, 0.5];              % 灰色

% --- (a) 理论Drude曲线（总群时延） ---
plot(f_theory(valid_theory)/1e9, tau_total_theory(valid_theory)*1e9, ...
    'Color', colors.theory, 'LineWidth', 2.5, ...
    'DisplayName', '理论Drude模型');
hold on;

% --- (b) FFT结果 (阶梯状+大误差) ---
plot(fft_f_probe/1e9, fft_tau*1e9, ...
    'Color', colors.fft, 'LineStyle', '--', 'LineWidth', 1.8, ...
    'Marker', 'x', 'MarkerSize', 10, ...
    'DisplayName', 'FFT提取 (频谱散焦)');

% --- (c) ESPRIT结果 (散点, 颜色表示权重) ---
% 归一化幅度权重
amp_norm = (feature_amplitude - min(feature_amplitude)) / ...
           (max(feature_amplitude) - min(feature_amplitude) + eps);

scatter(feature_f_probe/1e9, feature_tau*1e9, 50, amp_norm, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.3, ...
    'DisplayName', 'ESPRIT提取');
colormap(winter);
cb = colorbar;
ylabel(cb, '幅度权重 A_i (归一化)', 'FontName', 'SimHei', 'FontSize', 10);
caxis([0, 1]);

% --- 标注截止频率 ---
xline(f_c/1e9, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(f_c/1e9 + 0.08, 3.5, sprintf('f_p = %.0f GHz', f_c/1e9), ...
    'FontSize', 10, 'Rotation', 90, 'FontName', 'Times New Roman');

% --- 格式设置 ---
% 先设置坐标轴字体
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

% 再设置中文标签
xlabel('探测频率 f_{probe} / GHz', 'FontName', 'SimHei', 'FontSize', 12);
ylabel('总群时延 \tau_{total} / ns', 'FontName', 'SimHei', 'FontSize', 12);
title('特征轨迹重构对比：ESPRIT vs FFT', ...
    'FontName', 'SimHei', 'FontSize', 13, 'FontWeight', 'bold');

legend('Location', 'northwest', 'FontSize', 10, 'FontName', 'SimHei');
grid on;

xlim([34, 37.5]);
ylim([4.5, 5]);

% --- 参数注释框 ---
annotation('textbox', [0.62, 0.70, 0.30, 0.12], ...
    'String', {sprintf('参数: f_p = %.0f GHz, d = %.2f m', f_c/1e9, d), ...
               sprintf('窗口: T_w = 12 \\mus, 重叠率 90%%')}, ...
    'FontSize', 9, 'FontName', 'SimHei', ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', [1, 1, 0.95], ...
    'EdgeColor', [0.5, 0.5, 0.5]);

%% 9. 保存图表

output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

export_thesis_figure(gcf, '图4-6_特征轨迹重构对比', 14, 300, 'SimHei');

fprintf('\n图 4-6 已保存至 final_output/figures/\n');
fprintf('  - 图4-6_特征轨迹重构对比.png\n');
fprintf('  - 图4-6_特征轨迹重构对比.svg\n');

%% 10. 精度统计

fprintf('\n===== 精度对比统计 =====\n');

% 插值理论值到ESPRIT点
tau_theory_interp = interp1(f_theory(valid_theory), tau_total_theory(valid_theory), feature_f_probe);

% 过滤有效点
valid_stat = ~isnan(tau_theory_interp) & feature_tau > 0;
error_esprit = abs(feature_tau(valid_stat) - tau_theory_interp(valid_stat));
rmse_esprit = sqrt(mean(error_esprit.^2)) * 1e9;

fprintf('ESPRIT RMSE: %.4f ns\n', rmse_esprit);
fprintf('ESPRIT 最大误差: %.4f ns\n', max(error_esprit)*1e9);
fprintf('有效特征点数: %d\n', sum(valid_stat));

% FFT误差估计
tau_theory_fft = interp1(f_theory(valid_theory), tau_total_theory(valid_theory), fft_f_probe);
valid_fft_stat = ~isnan(tau_theory_fft);
error_fft = abs(fft_tau(valid_fft_stat) - tau_theory_fft(valid_fft_stat));
fprintf('FFT平均误差: %.4f ns (受频谱散焦影响)\n', mean(error_fft)*1e9);
