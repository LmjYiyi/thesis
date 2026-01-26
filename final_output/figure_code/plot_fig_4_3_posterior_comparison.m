%% plot_fig_4_3_posterior_comparison.m
% 论文图 4-3：参数可观测性的后验分布对比示意图（尖峰 vs 平原）
% 生成日期：2026-01-25
% 对应章节：4.1.3 反演策略假设：预设ν_e以实现参数降维的可行性论证
%
% 【重要更新】：本代码现在使用真实 MCMC 仿真数据，而非手工构造
% 基于 thesis-code/LM_MCMC_with_noise.m 的完整仿真流程
%
% 图表描述：
% "图4-3以示意图形式展示了这一参数可观测性的对比。如图所示，在同一坐标系中
%  绘制了两个概率密度函数：蓝色实线代表n_e的后验分布，呈现出针状的尖锐高斯
%  形态，峰值精准落在真值位置，标准差极小；红色虚线代表ν_e的后验分布，呈现
%  出桌面般的平坦均匀分布，覆盖整个先验范围[0.1, 10] GHz，没有任何向真值
%  聚集的趋势。"

clear; clc; close all;

fprintf('===================================================\n');
fprintf('图 4-3: 真实 MCMC 仿真生成后验分布对比图\n');
fprintf('===================================================\n');

%% ========================================================================
%  第一部分：LFMCW 等离子体仿真（简化版，保留核心物理）
% =========================================================================

fprintf('\n[1/4] 初始化仿真参数...\n');

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

fprintf('[2/4] 运行信号仿真...\n');

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

fprintf('[3/4] ESPRIT 特征提取...\n');

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

fprintf('[4/4] MCMC 采样生成后验分布...\n');

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

% MCMC 参数 (与论文4.3.2节一致)
N_samples = 10000;      % 总采样数
burn_in = 2000;         % 预烧期

% 先验分布范围 (与论文4.3.2节描述一致)
ne_min = 1e18; ne_max = 1e20;
nu_min = 0.1e9; nu_max = 5e9;   % 论文描述: [0.1, 5] GHz

% 提议分布步长
sigma_ne = (ne_max - ne_min) * 0.02;
sigma_nu = (nu_max - nu_min) * 0.03;

% 初始化
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
nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);

fprintf('  MCMC 采样完成 (接受率: %.1f%%)\n', accept_count/N_samples*100);
fprintf('  n_e 后验: 均值=%.3e, 标准差=%.2e (CV=%.2f%%)\n', ne_mean, ne_std, ne_std/ne_mean*100);
fprintf('  ν_e 后验: 均值=%.2f GHz, 标准差=%.2f GHz (CV=%.1f%%)\n', ...
        nu_mean/1e9, nu_std/1e9, nu_std/nu_mean*100);

%% ========================================================================
%  第五部分：绘制对比图
% =========================================================================

fprintf('\n绘制图 4-3...\n');

figure('Position', [100, 100, 1100, 500], 'Color', 'w');

% -------------------------------------------------------------------------
% 左图：n_e 后验分布（尖峰）
% -------------------------------------------------------------------------
subplot(1, 2, 1);

% 计算相对误差分布
ne_relative_error = (samples_ne_valid - n_e_true) / n_e_true * 100;

% 使用核密度估计平滑分布
[pdf_ne, xi_ne] = ksdensity(ne_relative_error, 'NumPoints', 500);
pdf_ne_norm = pdf_ne / max(pdf_ne);  % 归一化到最大值为1

% 填充区域
fill([xi_ne, fliplr(xi_ne)], [pdf_ne_norm, zeros(size(pdf_ne_norm))], ...
     [0.2, 0.5, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;

% 绘制曲线
plot(xi_ne, pdf_ne_norm, 'b-', 'LineWidth', 2.5);

% 标注真值位置
xline(0, 'k--', 'LineWidth', 1.5);
[~, peak_idx] = max(pdf_ne_norm);
plot(xi_ne(peak_idx), 1, 'b^', 'MarkerSize', 12, 'MarkerFaceColor', 'b');

% 标注 ±1σ 范围
cv_ne = ne_std / ne_mean * 100;
xline(-cv_ne, 'b:', 'LineWidth', 1.2);
xline(cv_ne, 'b:', 'LineWidth', 1.2);
text(cv_ne + 0.3, 0.6, sprintf('\sigma \approx %.1f%%', cv_ne), ...
     'FontSize', 11, 'FontName', 'SimHei', 'Color', 'b', 'Interpreter', 'tex');

% 动态调整X轴范围
x_range = max(abs(ne_relative_error)) * 1.2;
x_range = max(x_range, 5);  % 至少显示 ±5%
xlim([-x_range, x_range]);
ylim([0 1.15]);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

xlabel('\Delta n_e / n_e^{true} (%)', 'FontSize', 13, 'FontName', 'SimHei', 'Interpreter', 'tex');
ylabel('归一化后验概率密度', 'FontSize', 13, 'FontName', 'SimHei', 'Interpreter', 'tex');
title('(a) n_e 后验分布：尖锐高斯', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold', 'Interpreter', 'tex');

hold off;

% 添加说明文字
text(-x_range*0.9, 1.08, '\bf可观测', 'FontSize', 14, 'FontName', 'SimHei', 'Color', 'b', 'Interpreter', 'tex');
text(-x_range*0.9, 0.95, '数据提供强约束', 'FontSize', 10, 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);

% -------------------------------------------------------------------------
% 右图：ν_e 后验分布（平原）
% -------------------------------------------------------------------------
subplot(1, 2, 2);

% 转换为 GHz
nu_samples_GHz = samples_nu_valid / 1e9;
nu_true_GHz = nu_true / 1e9;

% 使用核密度估计 (范围与论文一致: 0.1-5 GHz)
[pdf_nu, xi_nu] = ksdensity(nu_samples_GHz, 'NumPoints', 500, ...
                            'Support', [0.1, 5], 'BoundaryCorrection', 'reflection');
pdf_nu_norm = pdf_nu / max(pdf_nu);

% 填充区域
fill([xi_nu, fliplr(xi_nu)], [pdf_nu_norm, zeros(size(pdf_nu_norm))], ...
     [0.8, 0.3, 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;

% 绘制曲线
plot(xi_nu, pdf_nu_norm, 'r--', 'LineWidth', 2.5);

% 标注真值位置
xline(nu_true_GHz, 'k--', 'LineWidth', 1.5);
[~, closest_idx] = min(abs(xi_nu - nu_true_GHz));
plot(nu_true_GHz, pdf_nu_norm(closest_idx), 'rv', 'MarkerSize', 12, 'LineWidth', 2);

% 标注先验边界 (与论文一致: 0.1-5 GHz)
xline(0.1, 'r:', 'LineWidth', 1.5);
xline(5, 'r:', 'LineWidth', 1.5);
text(0.3, 0.15, '先验下界', 'FontSize', 9, 'FontName', 'SimHei', 'Color', [0.6 0.2 0.2], 'Rotation', 90);
text(4.7, 0.15, '先验上界', 'FontSize', 9, 'FontName', 'SimHei', 'Color', [0.6 0.2 0.2], 'Rotation', 90);

xlim([0 5.5]);
ylim([0 1.15]);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

xlabel('\nu_e (GHz)', 'FontSize', 13, 'FontName', 'SimHei', 'Interpreter', 'tex');
ylabel('归一化后验概率密度', 'FontSize', 13, 'FontName', 'SimHei', 'Interpreter', 'tex');
title('(b) \nu_e 后验分布：平坦均匀', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold', 'Interpreter', 'tex');

hold off;

% 添加说明文字
text(0.5, 1.08, '\bf不可辨识', 'FontSize', 14, 'FontName', 'SimHei', 'Color', 'r', 'Interpreter', 'tex');
text(0.5, 0.95, '数据不提供约束', 'FontSize', 10, 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);

% 标注真值
text(nu_true_GHz + 0.3, pdf_nu_norm(closest_idx) + 0.08, '真值位置', ...
     'FontSize', 10, 'FontName', 'SimHei', 'Color', 'k');

%% 添加总标题
sgtitle('图 4-3 参数可观测性对比：尖峰 vs 平原 (MCMC真实仿真)', ...
        'FontSize', 16, 'FontName', 'SimHei', 'FontWeight', 'bold', 'Interpreter', 'tex');

%% ========================================================================
%  第六部分：保存图表与输出统计
% =========================================================================

% 确定保存路径
save_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

print('-dpng', '-r300', fullfile(save_dir, '图4-3_后验分布对比_尖峰vs平原.png'));
print('-dsvg', fullfile(save_dir, '图4-3_后验分布对比_尖峰vs平原.svg'));

fprintf('\n===================================================\n');
fprintf('图 4-3 已保存至 final_output/figures/\n');
fprintf('===================================================\n');
fprintf('\n【仿真参数】\n');
fprintf('  电子密度真值: n_e = %.4e m^-3\n', n_e_true);
fprintf('  碰撞频率真值: ν_e = %.2f GHz\n', nu_true/1e9);
fprintf('  信噪比: SNR = %d dB\n', SNR_dB);
fprintf('\n【后验分布统计】\n');
fprintf('  n_e: 后验均值 = %.4e, 标准差 = %.2e (CV = %.2f%%)\n', ne_mean, ne_std, ne_std/ne_mean*100);
fprintf('  ν_e: 后验均值 = %.2f GHz, 标准差 = %.2f GHz (CV = %.1f%%)\n', ...
        nu_mean/1e9, nu_std/1e9, nu_std/nu_mean*100);
cv_ne_val = ne_std/ne_mean*100;
cv_nu_val = nu_std/nu_mean*100;
fprintf('\n【结论】\n');
fprintf('  - n_e 变异系数 %.2f%% → %s\n', cv_ne_val, get_observability_label(cv_ne_val));
fprintf('  - ν_e 变异系数 %.1f%% → %s\n', cv_nu_val, get_observability_label(cv_nu_val));
fprintf('  - 这直观诠释了"不可辨识性"的概率定义\n');
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
