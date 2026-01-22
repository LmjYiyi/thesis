%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4-10 & 4-11: MCMC迹线图、后验分布与Corner Plot
% 
% 对应论文：第4章 4.4.3节 MCMC后验分布分析
% 
% 输出图表：
%   - Figure 4-10: MCMC迹线图与后验边缘分布 (2x2子图)
%   - Figure 4-11: 参数联合后验分布Corner Plot
%
% 依赖：需要先运行 LM_MCMC_with_noise.m 生成 MCMC 采样数据
%       或直接运行本脚本（内嵌完整的仿真和采样逻辑）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

fprintf('================================================\n');
fprintf(' Fig 4-10 & 4-11: MCMC后验分布分析\n');
fprintf('================================================\n\n');

%% 1. 仿真参数设置 (与表4-1完全一致)
f_start = 34.2e9;            
f_end = 37.4e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

tau_air = 4e-9;              
tau_fs = 1.75e-9;            
d = 150e-3;                  
f_c = 33e9;   % 截止频率 33 GHz (强色散)               
nu = 1.5e9;   % 碰撞频率 1.5 GHz              

c = 3e8;                     
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;

omega_p = 2*pi*f_c;          
n_e = (omega_p^2 * epsilon_0 * m_e) / e^2; % 真实电子密度

t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

f = (0:N-1)*(f_s/N);         
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

SNR_dB = 20;

fprintf('仿真参数:\n');
fprintf('  截止频率: f_p = %.0f GHz\n', f_c/1e9);
fprintf('  碰撞频率: nu_e = %.1f GHz\n', nu/1e9);
fprintf('  电子密度: n_e = %.2e m^-3\n', n_e);
fprintf('  信噪比: SNR = %d dB\n', SNR_dB);
fprintf('\n');

%% 2. LFMCW信号生成与传播
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
s_rx_plasma = s_rx_plasma + sqrt(Pn) * randn(size(s_rx_plasma));

% 混频与滤波
s_mix_plasma = s_tx .* real(s_rx_plasma);
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('信号仿真完成\n');

%% 3. ESPRIT特征提取 (简化版)
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
    
    num_sources = 1; % 简化：假设单信源
    
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

%% 4. MCMC参数反演
fprintf('\n开始MCMC采样...\n');

tau_relative_meas = feature_tau_absolute - tau_air;
fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (tau_relative_meas > 1e-11);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);
Weights = (W_raw / max(W_raw)).^2; 

sigma_meas = 0.1e-9;

% MCMC配置
N_samples = 10000;
burn_in = 2000;

ne_min = 1e18; ne_max = 1e20;
nu_min = 0.1e9; nu_max = 5e9;

sigma_ne = (ne_max - ne_min) * 0.02;
sigma_nu = (nu_max - nu_min) * 0.05;

rng(42);
ne_current = ne_min + (ne_max - ne_min) * rand();
nu_current = nu_min + (nu_max - nu_min) * rand();

logL_current = compute_log_likelihood_local(X_fit, Y_fit, Weights, ne_current, nu_current, ...
                                       sigma_meas, d, c, epsilon_0, m_e, e);

samples_ne = zeros(N_samples, 1);
samples_nu = zeros(N_samples, 1);
accept_count = 0;

hWait = waitbar(0, 'MCMC采样中...');
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
    
    if mod(i, 500) == 0
        waitbar(i/N_samples, hWait);
    end
end
close(hWait);

fprintf('MCMC完成: 接受率 = %.1f%%\n', accept_count/N_samples*100);

samples_ne_valid = samples_ne(burn_in+1:end);
samples_nu_valid = samples_nu(burn_in+1:end);

ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]);

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);

cv_ne = ne_std / ne_mean * 100;
cv_nu = nu_std / nu_mean * 100;

fprintf('\n后验统计:\n');
fprintf('  n_e: 均值=%.2e, CV=%.1f%%, 95%%CI=[%.2e, %.2e]\n', ne_mean, cv_ne, ne_ci(1), ne_ci(2));
fprintf('  nu_e: 均值=%.2f GHz, CV=%.1f%%\n', nu_mean/1e9, cv_nu);

%% 5. 绘制 Figure 4-10: MCMC迹线图与后验分布

figure('Name', 'Fig 4-10: MCMC迹线图与后验分布', 'Color', 'w', 'Position', [100 100 1000 700]);

% (a) n_e 迹线图
subplot(2,2,1);
plot(samples_ne, 'b', 'LineWidth', 0.5);
hold on;
yline(n_e, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'LineWidth', 1.5);
xlabel('迭代次数', 'FontSize', 11);
ylabel('n_e (m^{-3})', 'FontSize', 11);
title('(a) n_e 迹线图', 'FontSize', 12, 'FontWeight', 'bold');
legend('采样链', '真值', '预烧期边界', 'Location', 'best', 'FontSize', 9);
grid on; box on;

% (b) nu_e 迹线图
subplot(2,2,2);
plot(samples_nu/1e9, 'b', 'LineWidth', 0.5);
hold on;
yline(nu/1e9, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'LineWidth', 1.5);
xlabel('迭代次数', 'FontSize', 11);
ylabel('\nu_e (GHz)', 'FontSize', 11);
title('(b) \nu_e 迹线图', 'FontSize', 12, 'FontWeight', 'bold');
legend('采样链', '真值', '预烧期边界', 'Location', 'best', 'FontSize', 9);
grid on; box on;

% (c) n_e 后验分布
subplot(2,2,3);
histogram(samples_ne_valid, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
hold on;
xline(n_e, 'r--', 'LineWidth', 2.5);
xline(ne_ci(1), 'k--', 'LineWidth', 1); 
xline(ne_ci(2), 'k--', 'LineWidth', 1);
xlabel('n_e (m^{-3})', 'FontSize', 11);
ylabel('概率密度', 'FontSize', 11);
title(sprintf('(c) n_e 后验分布 (CV=%.1f%%)', cv_ne), 'FontSize', 12, 'FontWeight', 'bold');
legend('后验', '真值', '95% CI', 'Location', 'best', 'FontSize', 9);
grid on; box on;

% (d) nu_e 后验分布
subplot(2,2,4);
histogram(samples_nu_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'none');
hold on;
xline(nu/1e9, 'r--', 'LineWidth', 2.5);
xline(nu_ci(1)/1e9, 'k--', 'LineWidth', 1); 
xline(nu_ci(2)/1e9, 'k--', 'LineWidth', 1);
xlabel('\nu_e (GHz)', 'FontSize', 11);
ylabel('概率密度', 'FontSize', 11);
title(sprintf('(d) \\nu_e 后验分布 (CV=%.1f%%)', cv_nu), 'FontSize', 12, 'FontWeight', 'bold');
legend('后验', '真值', '95% CI', 'Location', 'best', 'FontSize', 9);
grid on; box on;

sgtitle(sprintf('图4-10 MCMC迹线图与后验边缘分布 (SNR=%d dB)', SNR_dB), 'FontSize', 14, 'FontWeight', 'bold');

% 保存
saveas(gcf, '../figures/图4-10_MCMC迹线图与后验分布.png');
print(gcf, '../figures/图4-10_MCMC迹线图与后验分布.pdf', '-dpdf', '-bestfit');
fprintf('\n图4-10已保存\n');

%% 6. 绘制 Figure 4-11: Corner Plot

figure('Name', 'Fig 4-11: Corner Plot', 'Color', 'w', 'Position', [150 150 700 650]);

% 主对角线: n_e边缘分布
subplot(2,2,1);
histogram(samples_ne_valid, 40, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
hold on;
xline(n_e, 'r--', 'LineWidth', 2);
xlabel('n_e (m^{-3})', 'FontSize', 11);
ylabel('PDF', 'FontSize', 11);
title('n_e 边缘分布', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% 主对角线: nu_e边缘分布
subplot(2,2,4);
histogram(samples_nu_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'none');
hold on;
xline(nu/1e9, 'r--', 'LineWidth', 2);
xlabel('\nu_e (GHz)', 'FontSize', 11);
ylabel('PDF', 'FontSize', 11);
title('\nu_e 边缘分布', 'FontSize', 11, 'FontWeight', 'bold');
grid on; box on;

% 下三角: 联合分布散点图
subplot(2,2,3);
downsample_idx = 1:5:length(samples_ne_valid);  % 降采样以提高可视性
scatter(samples_ne_valid(downsample_idx), samples_nu_valid(downsample_idx)/1e9, 8, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(n_e, nu/1e9, 'r+', 'MarkerSize', 18, 'LineWidth', 3);
xlabel('n_e (m^{-3})', 'FontSize', 11);
ylabel('\nu_e (GHz)', 'FontSize', 11);
title('联合后验分布', 'FontSize', 11, 'FontWeight', 'bold');
legend('后验样本', '真值', 'Location', 'best', 'FontSize', 9);
grid on; box on;

% 上三角: 相关系数
subplot(2,2,2);
corr_val = corrcoef(samples_ne_valid, samples_nu_valid);
text(0.5, 0.6, sprintf('皮尔逊相关系数', corr_val(1,2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
text(0.5, 0.4, sprintf('\\rho = %.3f', corr_val(1,2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
text(0.5, 0.2, '(接近零 → 参数解耦)', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.4 0.4 0.4]);
axis off;
title('参数耦合分析', 'FontSize', 11, 'FontWeight', 'bold');

sgtitle('图4-11 参数联合后验分布Corner Plot', 'FontSize', 14, 'FontWeight', 'bold');

% 保存
saveas(gcf, '../figures/图4-11_CornerPlot.png');
print(gcf, '../figures/图4-11_CornerPlot.pdf', '-dpdf', '-bestfit');
fprintf('图4-11已保存\n');

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
