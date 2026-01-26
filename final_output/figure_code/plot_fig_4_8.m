%% plot_fig_4_8.m
% 论文图 4-8：Corner Plot——参数可观测性与耦合结构可视化
% 生成日期：2026-01-25
% 对应章节：4.3.3 参数可观测性判据
%
% 【重要更新】：本代码使用真实 LFMCW 信号处理流程生成数据
% 基于 thesis-code/LM_MCMC_with_noise.m 的完整仿真流程
%
% 物理意义：
% - 边缘后验分布的CV（变异系数）反映参数可观测性
% - CV判据分类标准：
%     CV < 5%    ：强可观测（高精度、稳定反演）
%     5% - 15%   ：中等可观测（可信但精度受限）
%     15% - 30%  ：弱可观测（可被数据约束，但偏差显著）
%     CV > 30%   ：不可观测（由先验主导）
% - 联合分布呈"纵向长条"结构：n_e被数据约束在窄带内，ν_e在先验范围弥散

clear; clc; close all;

fprintf('===================================================\n');
fprintf('图 4-8: 真实 LFMCW 信号 Corner Plot\n');
fprintf('===================================================\n');

%% ========================================================================
%  第一部分：LFMCW 等离子体仿真参数设置
% =========================================================================

fprintf('\n[1/5] 初始化仿真参数...\n');

% LFMCW雷达参数
f_start = 34.2e9;            
f_end = 37.4e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

% 传播介质参数
tau_air = 4e-9;              
tau_fs = 1.75e-9;            
d = 150e-3;                  

% 【真实物理参数设定】
f_c = 33e9;                  % 等离子体截止频率 (对应 n_e ≈ 1.35e19 m^-3)
nu_true = 1.5e9;             % 碰撞频率真值 1.5 GHz

% 物理常数
c = 3e8;                     
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;

% 计算真实电子密度
omega_p = 2*pi*f_c;          
n_e_true = (omega_p^2 * epsilon_0 * m_e) / e^2;

fprintf('  电子密度真值: n_e = %.4e m^-3\n', n_e_true);
fprintf('  碰撞频率真值: ν_e = %.2f GHz\n', nu_true/1e9);

% 时间与频率轴
t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

f = (0:N-1)*(f_s/N);         
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

%% ========================================================================
%  第二部分：信号生成与传播仿真
% =========================================================================

fprintf('[2/5] 运行信号仿真...\n');

% LFMCW信号生成
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              

% 空气介质传播
delay_samples_air = round(tau_air/t_s);
s_rx_air = [zeros(1, delay_samples_air) s_tx(1:end-delay_samples_air)];

% 等离子体介质传播
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];
S_after_fs1 = fft(s_after_fs1);

% Drude模型传递函数
omega_safe = omega; 
omega_safe(omega_safe == 0) = 1e-10; 
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu_true);
epsilon_r_complex(omega == 0) = 1; 
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
k_real = real(k_complex);
k_imag = imag(k_complex);
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));
s_rx_plasma = [zeros(1, delay_samples_fs) s_after_plasma(1:end-delay_samples_fs)];

% 添加高斯白噪声 (SNR = 20dB)
SNR_dB = 20;
Ps = mean(s_rx_plasma.^2);
Pn = Ps / (10^(SNR_dB/10));
noise = sqrt(Pn) * randn(size(s_rx_plasma));
s_rx_plasma = s_rx_plasma + noise;

% 混频与低通滤波
s_mix_plasma = s_tx .* real(s_rx_plasma);
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('  信号仿真完成 (SNR = %d dB)\n', SNR_dB);

%% ========================================================================
%  第三部分：ESPRIT 特征提取
% =========================================================================

fprintf('[3/5] ESPRIT 特征提取...\n');

% 数据预处理
decimation_factor = 200; 
f_s_proc = f_s / decimation_factor; 
s_proc = s_if_plasma(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

% 算法参数
win_time = 12e-6;                
win_len = round(win_time * f_s_proc); 
step_len = round(win_len / 10);  
L_sub = round(win_len / 2);     

feature_f_probe = []; 
feature_tau_absolute = []; 
feature_amplitude = [];

% ESPRIT 处理循环
num_windows = floor((N_proc - win_len) / step_len) + 1;

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end
    
    x_window = s_proc(idx_start:idx_end);
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;
    
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % Hankel矩阵构建
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end
    
    % 前后向平均
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;
    
    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend'); 
    eig_vecs = eig_vecs(:, sort_idx);
    
    % MDL 准则
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
    num_sources = max(1, min(k_est, 3));
    
    % TLS-ESPRIT
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6); 
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    f_beat_est = min(valid_freqs); 
    amp_est = rms(x_window); 
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];
end

fprintf('  提取到 %d 个有效特征点\n', length(feature_f_probe));

%% ========================================================================
%  第四部分：MCMC 参数反演
% =========================================================================

fprintf('[4/5] MCMC 采样生成后验分布...\n');

% 数据准备
tau_relative_meas = feature_tau_absolute - tau_air;

fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (tau_relative_meas > 1e-11);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);

if isempty(X_fit)
    error('有效拟合数据点为空！');
end

Weights = (W_raw / max(W_raw)).^2; 
sigma_meas = 0.1e-9 * (1 + exp(-(SNR_dB-10)/5));

% MCMC 参数
N_samples = 10000;      % 总采样次数
burn_in = 2000;         % 预烧期

% 先验分布范围
ne_min = 1e18; ne_max = 1e20;
nu_min = 0.1e9; nu_max = 5e9;

% 提议分布步长
sigma_ne = (ne_max - ne_min) * 0.02;
sigma_nu = (nu_max - nu_min) * 0.05;

% 初始化 (从先验分布随机采样)
rng(42);
ne_current = ne_min + (ne_max - ne_min) * rand();
nu_current = nu_min + (nu_max - nu_min) * rand();

logL_current = compute_log_likelihood_local(X_fit, Y_fit, Weights, ne_current, nu_current, ...
                                            sigma_meas, d, c, epsilon_0, m_e, e);

samples_ne = zeros(N_samples, 1);
samples_nu = zeros(N_samples, 1);
accept_count = 0;

% MCMC 主循环
for i = 1:N_samples
    ne_proposed = ne_current + sigma_ne * randn();
    nu_proposed = nu_current + sigma_nu * randn();
    
    if ne_proposed < ne_min || ne_proposed > ne_max || ...
       nu_proposed < nu_min || nu_proposed > nu_max
        samples_ne(i) = ne_current;
        samples_nu(i) = nu_current;
        continue;
    end
    
    logL_proposed = compute_log_likelihood_local(X_fit, Y_fit, Weights, ne_proposed, nu_proposed, ...
                                                  sigma_meas, d, c, epsilon_0, m_e, e);
    
    log_alpha = logL_proposed - logL_current;
    
    if log(rand()) < log_alpha
        ne_current = ne_proposed;
        nu_current = nu_proposed;
        logL_current = logL_proposed;
        accept_count = accept_count + 1;
    end
    
    samples_ne(i) = ne_current;
    samples_nu(i) = nu_current;
end

% 丢弃预烧期
samples_ne_valid = samples_ne(burn_in+1:end);
samples_nu_valid = samples_nu(burn_in+1:end);

% 后验统计
ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]);
cv_ne = ne_std / ne_mean * 100;

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);
cv_nu = nu_std / nu_mean * 100;

% 相关系数
rho = corrcoef(samples_ne_valid, samples_nu_valid);
rho_val = rho(1, 2);

fprintf('  MCMC 采样完成 (接受率: %.1f%%)\n', accept_count/N_samples*100);
fprintf('  n_e 后验: 均值=%.3e, CV=%.2f%%\n', ne_mean, cv_ne);
fprintf('  ν_e 后验: 均值=%.2f GHz, CV=%.1f%%\n', nu_mean/1e9, cv_nu);
fprintf('  参数相关系数 ρ = %.3f\n', rho_val);

%% ========================================================================
%  第五部分：绘制图4-8 Corner Plot
% =========================================================================

fprintf('[5/5] 绘制图 4-8...\n');

figure('Position', [100, 100, 700, 700], 'Color', 'w');

% 标准颜色方案
color_ne = [0.2, 0.6, 0.8];     % 蓝色系
color_nu = [0.8, 0.4, 0.2];     % 橙色系
color_true = [0.8, 0.2, 0.2];   % 红色（真值）
color_scatter = [0.3, 0.5, 0.7];% 散点颜色

%% 子图(1,1): n_e 边缘后验分布
subplot(2, 2, 1);
histogram(samples_ne_valid, 50, 'Normalization', 'pdf', ...
    'FaceColor', color_ne, 'EdgeColor', 'w', 'FaceAlpha', 0.8);
hold on;

% 真值参考线
xline(n_e_true, '--', 'Color', color_true, 'LineWidth', 2.5);

% 95% CI 标注
xline(ne_ci(1), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);
xline(ne_ci(2), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

xlabel('$n_e\\;\\mathrm{(m^{-3})}$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('概率密度', 'FontSize', 12, 'FontName', 'SimHei');
title(sprintf('n_e 边缘后验 (CV = %.1f%%)', cv_ne), 'FontSize', 12, 'FontName', 'SimHei');

% 添加"尖锐窄峰"标注
text(ne_mean, max(ylim)*0.85, '← 尖锐窄峰', ...
    'FontSize', 10, 'Color', color_ne, 'FontWeight', 'bold', 'FontName', 'SimHei');

grid on; box on;

%% 子图(2,2): ν_e 边缘后验分布
subplot(2, 2, 4);
histogram(samples_nu_valid/1e9, 50, 'Normalization', 'pdf', ...
    'FaceColor', color_nu, 'EdgeColor', 'w', 'FaceAlpha', 0.8);
hold on;

% 真值参考线
xline(nu_true/1e9, '--', 'Color', color_true, 'LineWidth', 2.5);

% 先验范围标注
xline(nu_min/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
xline(nu_max/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

xlabel('$\\nu_e\\;\\mathrm{(GHz)}$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('概率密度', 'FontSize', 12, 'FontName', 'SimHei');
title(sprintf('nu_e 边缘后验 (CV = %.1f%%)', cv_nu), 'FontSize', 12, 'FontName', 'SimHei');

% 添加"平坦分布"标注
text(mean([nu_min, nu_max])/1e9, max(ylim)*0.85, '← 类均匀分布 →', ...
    'FontSize', 10, 'Color', color_nu, 'FontWeight', 'bold', 'FontName', 'SimHei', ...
    'HorizontalAlignment', 'center');

grid on; box on;

%% 子图(2,1): 联合后验分布（核心：纵向长条结构）
subplot(2, 2, 3);

% 降采样绘制散点图（提高可读性）
N_valid = length(samples_ne_valid);
subsample_idx = 1:5:N_valid;
scatter(samples_ne_valid(subsample_idx), samples_nu_valid(subsample_idx)/1e9, ...
    8, color_scatter, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

% 真值标注
plot(n_e_true, nu_true/1e9, 'r+', 'MarkerSize', 18, 'LineWidth', 3);

% 强调"纵向长条"结构
% 用虚线框标注n_e的约束范围
rectangle('Position', [ne_ci(1), nu_min/1e9, ne_ci(2)-ne_ci(1), (nu_max-nu_min)/1e9], ...
    'EdgeColor', [0.8, 0.2, 0.2], 'LineWidth', 2, 'LineStyle', '--');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

xlabel('$n_e\\;\\mathrm{(m^{-3})}$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('$\\nu_e\\;\\mathrm{(GHz)}$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
title('联合后验分布', 'FontSize', 12, 'FontName', 'SimHei');
grid on; box on;

% 添加结构标注
annotation('textarrow', [0.28, 0.32], [0.25, 0.35], 'String', '纵向长条结构', ...
    'FontSize', 11, 'FontName', 'SimHei', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 1.5);

% 标注物理意义
text(ne_mean*0.92, nu_max/1e9*0.9, ...
    sprintf('n_e 被数据约束\n\\nu_e 在先验内弥散'), ...
    'FontSize', 10, 'FontName', 'SimHei', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'HorizontalAlignment', 'center');

%% 子图(1,2): 参数耦合分析（相关系数）
subplot(2, 2, 2);
axis off;

% 绘制参数耦合信息
text(0.5, 0.75, '参数相关分析', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei', ...
    'HorizontalAlignment', 'center');

text(0.5, 0.55, sprintf('$\\rho_{n_e,\\nu_e} = %.3f$', rho_val), ...
    'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times New Roman', ...
    'HorizontalAlignment', 'center');

if abs(rho_val) < 0.3
    coupling_desc = '弱耦合（可独立反演）';
elseif abs(rho_val) < 0.7
    coupling_desc = '中等耦合';
else
    coupling_desc = '强耦合（需重新参数化）';
end
text(0.5, 0.38, coupling_desc, ...
    'FontSize', 12, 'FontName', 'SimHei', ...
    'HorizontalAlignment', 'center', 'Color', [0.3, 0.6, 0.3]);

% 添加CV判据说明框（动态分类）
ne_obs_label = get_observability_label(cv_ne);
nu_obs_label = get_observability_label(cv_nu);
text(0.5, 0.15, sprintf('CV判据：\nn_e: %.1f%% → %s\n\\nu_e: %.1f%% → %s', ...
    cv_ne, ne_obs_label, cv_nu, nu_obs_label), ...
    'Interpreter', 'tex', 'FontSize', 10, 'FontName', 'SimHei', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'k', 'Margin', 5);

%% 添加总标题
sgtitle(sprintf('图  Corner Plot：参数可观测性分析 (真实LFMCW信号, SNR=%ddB)', SNR_dB), ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

%% ========================================================================
%  第六部分：保存图表
% =========================================================================

save_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

print('-dpng', '-r300', fullfile(save_dir, '图4-8_Corner_Plot.png'));
print('-dsvg', fullfile(save_dir, '图4-8_Corner_Plot.svg'));

fprintf('\n===================================================\n');
fprintf('图 4-8 已保存至 final_output/figures/\n');
fprintf('===================================================\n');
fprintf('\n【仿真参数】\n');
fprintf('  电子密度真值: n_e = %.4e m^-3\n', n_e_true);
fprintf('  碰撞频率真值: ν_e = %.2f GHz\n', nu_true/1e9);
fprintf('  信噪比: SNR = %d dB\n', SNR_dB);
fprintf('  MCMC采样数: %d, 预烧期: %d\n', N_samples, burn_in);
fprintf('\n【后验分布统计】\n');
fprintf('  n_e: CV = %.2f%% → %s\n', cv_ne, get_observability_label(cv_ne));
fprintf('  ν_e: CV = %.1f%% → %s\n', cv_nu, get_observability_label(cv_nu));
fprintf('  参数相关系数 ρ = %.3f\n', rho_val);
fprintf('\n【CV判据分类标准】\n');
fprintf('  CV < 5%%    ：强可观测\n');
fprintf('  5%% - 15%%  ：中等可观测\n');
fprintf('  15%% - 30%% ：弱可观测\n');
fprintf('  CV > 30%%   ：不可观测\n');

%% ========================================================================
%  局部函数
% =========================================================================

function logL = compute_log_likelihood_local(f_data, tau_data, weights, ne_val, nu_val, sigma, d, c, eps0, me, e_charge)
    if ne_val <= 0
        logL = -1e10; return;
    end
    
    fc_val = sqrt(ne_val * e_charge^2 / (eps0 * me)) / (2*pi);
    
    if fc_val >= (min(f_data) - 0.05e9)
        logL = -1e10; return;
    end
    
    try
        tau_theory = calculate_theoretical_delay_local(f_data, ne_val, nu_val, d, c, eps0, me, e_charge);
        residuals = (tau_theory - tau_data) / sigma;
        logL = -0.5 * sum(weights .* residuals.^2);
        
        if isnan(logL) || isinf(logL)
            logL = -1e10;
        end
    catch
        logL = -1e10;
    end
end

function tau_rel = calculate_theoretical_delay_local(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
    omega_vec = 2 * pi * f_vec;
    wp_val = sqrt(ne_val * e_charge^2 / (eps0 * me));
    
    eps_r = 1 - (wp_val^2) ./ (omega_vec .* (omega_vec + 1i*nu_val));
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    phi_plasma = -real(k_vec) * d;
    
    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    
    tau_total = -d_phi ./ d_omega;
    tau_total = [tau_total, tau_total(end)];
    
    tau_rel = tau_total - (d/c);
end

function label = get_observability_label(cv_value)
    % 根据CV值返回可观测性等级标签
    % CV判据分类标准：
    %   CV < 5%    ：强可观测（高精度、稳定反演）
    %   5% - 15%   ：中等可观测（可信但精度受限）
    %   15% - 30%  ：弱可观测（可被数据约束，但偏差显著）
    %   CV > 30%   ：不可观测（由先验主导）
    
    if cv_value < 5
        label = '强可观测';
    elseif cv_value < 15
        label = '中等可观测';
    elseif cv_value < 30
        label = '弱可观测';
    else
        label = '不可观测';
    end
end
