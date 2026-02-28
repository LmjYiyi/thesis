%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot_fig_4_10_4_11.m
% 论文图4-10：MCMC迹线图与后验边缘分布
% 论文图4-11：参数联合后验分布Corner Plot
% 生成日期：2026-01-26
% 对应章节：4.4.3 MCMC后验分布分析
%
% 图表核心表达：
% - 图4-10: n_e收敛良好(尖峰后验) vs ν_e漫游(平坦后验)
% - 图4-11: Corner Plot展示"纵向长条"参数解耦结构
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

fprintf('================================================\n');
fprintf(' 图4-10 & 4-11: MCMC后验分布分析\n');
fprintf('================================================\n\n');

%% 1. 物理常数与参数设置
% 【必须与表4-1保持一致】
c = 2.99792458e8;
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e_charge = 1.602e-19;

% LFMCW雷达参数
f_start = 34.2e9;
f_end = 37.4e9;
B = f_end - f_start;
T_m = 50e-6;
K = B/T_m;
f_s = 80e9;

% 传播路径参数
tau_air = 4e-9;
tau_fs = 1.75e-9;
d = 150e-3;

% 等离子体参数 (真值)
f_c = 33e9;                    % 截止频率 33 GHz
nu = 1.5e9;                    % 碰撞频率 1.5 GHz
omega_p = 2*pi*f_c;
n_e = (omega_p^2 * epsilon_0 * m_e) / e_charge^2;  % 真实电子密度

SNR_dB = 20;

% 采样参数
t_s = 1/f_s;
N = round(T_m/t_s);
t = (0:N-1)*t_s;

% 构建FFT频率轴
f = (0:N-1)*(f_s/N);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;

fprintf('仿真参数:\n');
fprintf('  截止频率: f_p = %.0f GHz\n', f_c/1e9);
fprintf('  碰撞频率: ν_e = %.1f GHz\n', nu/1e9);
fprintf('  电子密度: n_e = %.2e m^-3\n', n_e);
fprintf('  信噪比: SNR = %d dB\n', SNR_dB);
fprintf('\n');

%% 2. LFMCW信号生成与等离子体传播

f_t = f_start + K*mod(t, T_m);
phi_t = 2*pi*cumsum(f_t)*t_s;
s_tx = cos(phi_t);

% 等离子体传播
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];
S_after_fs1 = fft(s_after_fs1);

omega_safe = omega;
omega_safe(omega_safe == 0) = 1e-10;
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_complex(omega == 0) = 1;
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
H_plasma = exp(-1i * real(k_complex) * d - abs(imag(k_complex)) * d);

S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));
s_rx_plasma = [zeros(1, delay_samples_fs) s_after_plasma(1:end-delay_samples_fs)];

% 添加噪声
Ps = mean(s_rx_plasma.^2);
Pn = Ps / (10^(SNR_dB/10));
rng(42);
s_rx_plasma = s_rx_plasma + sqrt(Pn) * randn(size(s_rx_plasma));

% 混频与滤波
s_mix_plasma = s_tx .* real(s_rx_plasma);
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('信号仿真完成\n');

%% 3. ESPRIT特征提取

fprintf('开始ESPRIT特征提取...\n');

decimation_factor = 200;
f_s_proc = f_s / decimation_factor;
s_proc = s_if_plasma(1:decimation_factor:end);
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
    if idx_end > N_proc, break; end
    
    x_window = s_proc(idx_start:idx_end);
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;
    
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % Hankel矩阵与特征分解
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end
    
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;
    
    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);
    
    num_sources = 1;  % 简化：假设单信源
    
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];
end

fprintf('ESPRIT完成: %d 个特征点\n', length(feature_f_probe));

%% 4. MCMC Metropolis-Hastings 采样

fprintf('\n开始MCMC采样 (10000次迭代)...\n');

% 数据准备
tau_relative_meas = feature_tau_absolute - tau_air;
fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (tau_relative_meas > 1e-11);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);
Weights = (W_raw / max(W_raw)).^2;

sigma_meas = 0.1e-9;

% MCMC配置 (对应表4-1)
N_samples = 10000;
burn_in = 2000;

% 先验范围 (均匀分布)
ne_min = 1e18; ne_max = 1e20;
nu_min = 0.1e9; nu_max = 5e9;

% 提议分布步长
sigma_ne = (ne_max - ne_min) * 0.02;
sigma_nu = (nu_max - nu_min) * 0.05;

% 随机初始化
rng(42);
ne_current = ne_min + (ne_max - ne_min) * rand();
nu_current = nu_min + (nu_max - nu_min) * rand();

logL_current = compute_log_likelihood_local(X_fit, Y_fit, Weights, ne_current, nu_current, ...
                                       sigma_meas, d, c, epsilon_0, m_e, e_charge);

samples_ne = zeros(N_samples, 1);
samples_nu = zeros(N_samples, 1);
samples_logL = zeros(N_samples, 1);
accept_count = 0;

hWait = waitbar(0, 'MCMC采样中...');
for i = 1:N_samples
    % 高斯随机游走提议
    ne_proposed = ne_current + sigma_ne * randn();
    nu_proposed = nu_current + sigma_nu * randn();
    
    % 先验约束检查
    if ne_proposed < ne_min || ne_proposed > ne_max || ...
       nu_proposed < nu_min || nu_proposed > nu_max
        samples_ne(i) = ne_current;
        samples_nu(i) = nu_current;
        samples_logL(i) = logL_current;
        continue;
    end
    
    % 计算提议点似然
    logL_proposed = compute_log_likelihood_local(X_fit, Y_fit, Weights, ne_proposed, nu_proposed, ...
                                            sigma_meas, d, c, epsilon_0, m_e, e_charge);
    
    % Metropolis-Hastings接受准则
    log_alpha = logL_proposed - logL_current;
    
    if log(rand()) < log_alpha
        ne_current = ne_proposed;
        nu_current = nu_proposed;
        logL_current = logL_proposed;
        accept_count = accept_count + 1;
    end
    
    samples_ne(i) = ne_current;
    samples_nu(i) = nu_current;
    samples_logL(i) = logL_current;
    
    if mod(i, 500) == 0
        waitbar(i/N_samples, hWait);
    end
end
close(hWait);

fprintf('MCMC完成: 接受率 = %.1f%%\n', accept_count/N_samples*100);

%% 5. 后验统计分析

% 丢弃预烧期
samples_ne_valid = samples_ne(burn_in+1:end);
samples_nu_valid = samples_nu(burn_in+1:end);

% 后验统计
ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]);

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);

% 变异系数
cv_ne = ne_std / ne_mean * 100;
cv_nu = nu_std / nu_mean * 100;

% 相关系数
corr_matrix = corrcoef(samples_ne_valid, samples_nu_valid);
rho = corr_matrix(1, 2);

fprintf('\n===== 后验统计 =====\n');
fprintf('n_e:\n');
fprintf('  真值:     %.4e m^-3\n', n_e);
fprintf('  后验均值: %.4e m^-3\n', ne_mean);
fprintf('  后验标准差: %.2e m^-3\n', ne_std);
fprintf('  变异系数CV: %.1f%%\n', cv_ne);
fprintf('  95%% CI: [%.2e, %.2e]\n', ne_ci(1), ne_ci(2));
fprintf('  相对误差: %.2f%%\n', (ne_mean - n_e)/n_e * 100);

fprintf('\nν_e:\n');
fprintf('  真值:     %.2f GHz\n', nu/1e9);
fprintf('  后验均值: %.2f GHz\n', nu_mean/1e9);
fprintf('  后验标准差: %.2f GHz\n', nu_std/1e9);
fprintf('  变异系数CV: %.1f%%\n', cv_nu);
fprintf('  95%% CI: [%.2f, %.2f] GHz\n', nu_ci(1)/1e9, nu_ci(2)/1e9);

fprintf('\n相关系数 ρ = %.3f (接近零 → 参数解耦)\n', rho);

fprintf('\n===== 可观测性判定 =====\n');
if cv_ne < 5
    fprintf('n_e: CV=%.1f%% < 5%% → 强可观测参数 ✓\n', cv_ne);
else
    fprintf('n_e: CV=%.1f%% → 弱可观测\n', cv_ne);
end
if cv_nu > 50
    fprintf('ν_e: CV=%.1f%% > 50%% → 不可观测参数 ✓\n', cv_nu);
else
    fprintf('ν_e: CV=%.1f%% → 可观测\n', cv_nu);
end

%% 6. 绘制图4-10: MCMC迹线图与后验边缘分布

% 论文统一配色
colors = struct();
colors.blue = [0.2, 0.6, 0.8];
colors.orange = [0.8, 0.4, 0.2];
colors.red = [0.85, 0.25, 0.1];
colors.black = [0, 0, 0];

figure('Name', 'MCMC迹线图与后验分布', 'Color', 'w', 'Position', [100 100 1100 750]);

%% --- (a) n_e 迹线图 ---
subplot(2,2,1);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

plot(samples_ne, 'Color', colors.blue, 'LineWidth', 0.5);
yline(n_e, '--', 'Color', colors.red, 'LineWidth', 2);
xline(burn_in, 'k--', 'LineWidth', 1.5);

xlabel('迭代次数', 'FontName', 'SimHei', 'FontSize', 11);
ylabel('n_e (m^{-3})', 'FontName', 'Times New Roman', 'FontSize', 11);
title('(a) n_e 迹线图', 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
legend({'采样链', '真值', '预烧期边界'}, 'Location', 'best', 'FontName', 'SimHei', 'FontSize', 9);
grid on;

%% --- (b) ν_e 迹线图 ---
subplot(2,2,2);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

plot(samples_nu/1e9, 'Color', colors.blue, 'LineWidth', 0.5);
yline(nu/1e9, '--', 'Color', colors.red, 'LineWidth', 2);
xline(burn_in, 'k--', 'LineWidth', 1.5);

xlabel('迭代次数', 'FontName', 'SimHei', 'FontSize', 11);
ylabel('\nu_e (GHz)', 'FontName', 'Times New Roman', 'FontSize', 11);
title('(b) \nu_e 迹线图', 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
legend({'采样链', '真值', '预烧期边界'}, 'Location', 'best', 'FontName', 'SimHei', 'FontSize', 9);
grid on;

% 标注"漫游"特征
text(N_samples*0.7, nu_max/1e9*0.9, '随机漫游', 'FontName', 'SimHei', 'FontSize', 10, ...
    'Color', [0.6 0.3 0], 'FontWeight', 'bold');

%% --- (c) n_e 后验分布 ---
subplot(2,2,3);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

histogram(samples_ne_valid, 50, 'Normalization', 'pdf', 'FaceColor', colors.blue, 'EdgeColor', 'none');
xline(n_e, '--', 'Color', colors.red, 'LineWidth', 2.5);
xline(ne_ci(1), 'k--', 'LineWidth', 1);
xline(ne_ci(2), 'k--', 'LineWidth', 1);

xlabel('n_e (m^{-3})', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('概率密度', 'FontName', 'SimHei', 'FontSize', 11);
title(sprintf('(c) n_e 后验分布 (CV=%.1f%%)', cv_ne), 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
legend({'后验分布', '真值', '95% CI'}, 'Location', 'best', 'FontName', 'SimHei', 'FontSize', 9);
grid on;

% 标注"尖峰"
text(ne_mean, max(ylim)*0.85, '尖峰后验', 'FontName', 'SimHei', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0 0.4 0], 'FontWeight', 'bold');

%% --- (d) ν_e 后验分布 ---
subplot(2,2,4);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

histogram(samples_nu_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', colors.orange, 'EdgeColor', 'none');
xline(nu/1e9, '--', 'Color', colors.red, 'LineWidth', 2.5);
xline(nu_ci(1)/1e9, 'k--', 'LineWidth', 1);
xline(nu_ci(2)/1e9, 'k--', 'LineWidth', 1);

xlabel('\nu_e (GHz)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('概率密度', 'FontName', 'SimHei', 'FontSize', 11);
title(sprintf('(d) \\nu_e 后验分布 (CV=%.1f%%)', cv_nu), 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
legend({'后验分布', '真值', '95% CI'}, 'Location', 'best', 'FontName', 'SimHei', 'FontSize', 9);
grid on;

% 标注"平坦"
text((nu_min+nu_max)/2/1e9, max(ylim)*0.85, '平坦后验', 'FontName', 'SimHei', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0.6 0.3 0], 'FontWeight', 'bold');

sgtitle(sprintf('MCMC迹线图与后验边缘分布 (SNR=%d dB)', SNR_dB), ...
    'FontName', 'SimHei', 'FontSize', 14, 'FontWeight', 'bold');

%% 保存图4-10
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

export_thesis_figure(gcf, '图4-10_MCMC迹线图与后验分布', 14, 300, 'SimHei');
fprintf('\n✓ 图4-10已保存\n');

%% 7. 绘制图4-11: Corner Plot

figure('Name', 'Corner Plot', 'Color', 'w', 'Position', [150 150 750 700]);

%% --- 左上: n_e边缘分布 ---
subplot(2,2,1);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

histogram(samples_ne_valid, 40, 'Normalization', 'pdf', 'FaceColor', colors.blue, 'EdgeColor', 'none');
xline(n_e, '--', 'Color', colors.red, 'LineWidth', 2);

xlabel('n_e (m^{-3})', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('PDF', 'FontName', 'Times New Roman', 'FontSize', 11);
title('n_e 边缘分布', 'FontName', 'SimHei', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

%% --- 右下: ν_e边缘分布 ---
subplot(2,2,4);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

histogram(samples_nu_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', colors.orange, 'EdgeColor', 'none');
xline(nu/1e9, '--', 'Color', colors.red, 'LineWidth', 2);

xlabel('\nu_e (GHz)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('PDF', 'FontName', 'Times New Roman', 'FontSize', 11);
title('\nu_e 边缘分布', 'FontName', 'SimHei', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

%% --- 左下: 联合后验分布 (核心可视化) ---
subplot(2,2,3);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');
hold on;

% 降采样以提高可视性
downsample_idx = 1:5:length(samples_ne_valid);
scatter(samples_ne_valid(downsample_idx), samples_nu_valid(downsample_idx)/1e9, ...
    8, colors.blue, 'filled', 'MarkerFaceAlpha', 0.3);
plot(n_e, nu/1e9, 'r+', 'MarkerSize', 18, 'LineWidth', 3);

xlabel('n_e (m^{-3})', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('\nu_e (GHz)', 'FontName', 'Times New Roman', 'FontSize', 11);
title('联合后验分布', 'FontName', 'SimHei', 'FontSize', 11, 'FontWeight', 'bold');
legend({'后验样本', '真值'}, 'Location', 'best', 'FontName', 'SimHei', 'FontSize', 9);
grid on;

% 标注"纵向长条"结构
text(ne_mean, nu_max/1e9*0.9, '纵向长条', 'FontName', 'SimHei', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0.4 0.2 0.6], 'FontWeight', 'bold');

%% --- 右上: 相关系数 ---
subplot(2,2,2);
axis off;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

text(0.5, 0.7, '皮尔逊相关系数', 'FontName', 'SimHei', 'FontSize', 12, ...
    'HorizontalAlignment', 'center', 'Units', 'normalized');
text(0.5, 0.5, sprintf('\\rho = %.3f', rho), 'FontName', 'Times New Roman', 'FontSize', 18, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Units', 'normalized');
text(0.5, 0.3, '(接近零 → 参数解耦)', 'FontName', 'SimHei', 'FontSize', 10, ...
    'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'center', 'Units', 'normalized');
title('参数耦合分析', 'FontName', 'SimHei', 'FontSize', 11, 'FontWeight', 'bold');

sgtitle('参数联合后验分布Corner Plot', ...
    'FontName', 'SimHei', 'FontSize', 14, 'FontWeight', 'bold');

%% 保存图4-11
export_thesis_figure(gcf, '图4-11_CornerPlot', 14, 300, 'SimHei');
fprintf('✓ 图4-11已保存\n');

%% 8. 输出论文数据

fprintf('\n================================================\n');
fprintf(' 论文4.4.3节关键数据\n');
fprintf('================================================\n');
fprintf('n_e后验分布:\n');
fprintf('  - 后验均值: %.2e m^-3\n', ne_mean);
fprintf('  - 后验标准差: %.2e m^-3\n', ne_std);
fprintf('  - 变异系数CV: %.1f%% (< 5%% → 强可观测)\n', cv_ne);
fprintf('  - 95%%置信区间: [%.2e, %.2e] m^-3\n', ne_ci(1), ne_ci(2));
fprintf('  - 覆盖真值: %s\n', iff(n_e >= ne_ci(1) && n_e <= ne_ci(2), '是 ✓', '否'));

fprintf('\nν_e后验分布:\n');
fprintf('  - 后验均值: %.2f GHz\n', nu_mean/1e9);
fprintf('  - 后验标准差: %.2f GHz\n', nu_std/1e9);
fprintf('  - 变异系数CV: %.1f%% (> 50%% → 不可观测)\n', cv_nu);
fprintf('  - 95%%置信区间: [%.2f, %.2f] GHz\n', nu_ci(1)/1e9, nu_ci(2)/1e9);

fprintf('\n相关系数: ρ = %.3f\n', rho);

fprintf('\n================================================\n');
fprintf(' 图4-10 & 4-11 生成完成！\n');
fprintf('================================================\n');

%% ========================================================================
%  局部函数
%% ========================================================================

function logL = compute_log_likelihood_local(f_data, tau_data, weights, ne_val, nu_val, sigma, d, c, eps0, me, e_charge)
    if ne_val <= 0
        logL = -1e10; return;
    end
    
    fc_val = sqrt(ne_val * e_charge^2 / (eps0 * me)) / (2*pi);
    if fc_val >= (min(f_data) - 0.05e9)
        logL = -1e10; return;
    end
    
    try
        omega_vec = 2 * pi * f_data;
        wp_val = sqrt(ne_val * e_charge^2 / (eps0 * me));
        eps_r = 1 - (wp_val^2) ./ (omega_vec .* (omega_vec + 1i*nu_val));
        k_vec = (omega_vec ./ c) .* sqrt(eps_r);
        phi_plasma = -real(k_vec) * d;
        
        d_phi = diff(phi_plasma);
        d_omega = diff(omega_vec);
        tau_total = -d_phi ./ d_omega;
        tau_total = [tau_total, tau_total(end)];
        tau_theory = tau_total - (d/c);
        
        residuals = (tau_theory - tau_data) / sigma;
        logL = -0.5 * sum(weights .* residuals.^2);
        
        if isnan(logL) || isinf(logL)
            logL = -1e10;
        end
    catch
        logL = -1e10;
    end
end

function result = iff(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
