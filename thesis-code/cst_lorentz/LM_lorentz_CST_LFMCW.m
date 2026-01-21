%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW超材料诊断系统 - 基于CST全波仿真数据
% 功能: 将LFMCW信号通过CST仿真的S21传递函数，实现真实全波仿真
% 数据来源: CST Microwave Studio 全波仿真 (CSRR-loaded WR-28 波导)
%
% 与LM_lorentz_MCMC.m的区别:
%   - 原脚本使用理论Lorentz模型计算传递函数
%   - 本脚本使用CST全波仿真的S21作为真实传递函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 仿真参数设置
clc; clear all; close all;

% =========================================================================
% LFMCW雷达参数 (与原脚本保持一致)
% =========================================================================
f_start = 34.2e9;            
f_end = 37.4e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

% 传播介质参数 (来自CST模型)
tau_air = 4e-9;              % 空气总时延
tau_fs = 1.75e-9;            % 单侧自由空间时延
d = 3e-3;                    % 有效厚度 (m) - 与CST模型一致!

% 计算派生参数
c = 3e8;                     
t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% --- 构建正确的FFT频率轴 (包含负频率) ---
f = (0:N-1)*(f_s/N);         
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

fprintf('========================================\n');
fprintf('LFMCW + CST全波仿真诊断系统\n');
fprintf('========================================\n');
fprintf('LFMCW频率范围: %.2f - %.2f GHz\n', f_start/1e9, f_end/1e9);
fprintf('有效厚度 d = %.1f mm (来自CST模型)\n', d*1e3);
fprintf('========================================\n\n');

%% 2. 读取CST S参数 (.s2p文件)
fprintf('正在读取CST S参数数据...\n');

s2p_file = fullfile(fileparts(mfilename('fullpath')),  'data', 'data.s2p');

% 读取s2p文件
[f_cst, S11, S21, S12, S22] = read_s2p_touchstone(s2p_file);

fprintf('✓ S参数读取完成: %d 个频率点\n', length(f_cst));
fprintf('  频率范围: %.2f - %.2f GHz\n', min(f_cst)/1e9, max(f_cst)/1e9);

% 检查频率覆盖
if min(f_cst) > f_start || max(f_cst) < f_end
    warning('CST频率范围 [%.1f, %.1f] GHz 未完全覆盖 LFMCW 频率 [%.1f, %.1f] GHz!', ...
        min(f_cst)/1e9, max(f_cst)/1e9, f_start/1e9, f_end/1e9);
end

%% 3. LFMCW信号生成模块
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              
fprintf('✓ LFMCW信号生成完成\n');

%% 4. 信号传播模拟模块 (使用CST S21传递函数)

% 4.1 空气介质传播模拟 (时延)
delay_samples_air = round(tau_air/t_s);
s_rx_air = [zeros(1, delay_samples_air) s_tx(1:end-delay_samples_air)];

% 4.2 超材料介质传播模拟 (使用CST S21)
fprintf('正在通过CST S21传递函数处理信号...\n');

% 第一段:自由空间
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];

% 第二段:穿过超材料 (频域处理 - 使用CST S21)
S_after_fs1 = fft(s_after_fs1);

% 构建完整频率轴的S21传递函数
% 使用简单的逐点插值方法，避免维度问题
H_cst = ones(1, N);  % 默认透明

for idx = 1:N
    f_point = f(idx);
    f_abs = abs(f_point);  % 使用绝对频率查找
    
    % 只处理CST覆盖范围内的频率
    if f_abs >= min(f_cst) && f_abs <= max(f_cst)
        S21_val = interp1(f_cst, S21, f_abs, 'linear');
        if ~isnan(S21_val)
            if f_point >= 0
                H_cst(idx) = S21_val;
            else
                H_cst(idx) = conj(S21_val);  % 负频率取共轭
            end
        end
    end
end


% 应用传递函数
S_after_meta = S_after_fs1 .* H_cst;
S_RX_meta_fft = S_after_meta; 

% 转回时域
s_after_meta = real(ifft(S_after_meta));

% 第三段:自由空间
s_rx_meta = [zeros(1, delay_samples_fs) s_after_meta(1:end-delay_samples_fs)];

fprintf('✓ CST传递函数处理完成\n');

%% 5. 混频处理与差频信号提取

% 5.1 空气介质混频
s_mix_air = s_tx .* s_rx_air;

% 低通滤波器设计
fc_lp = 100e6;  
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));

s_if_air = filtfilt(b_lp, a_lp, s_mix_air);

% 5.2 超材料介质混频
s_mix_meta = s_tx .* real(s_rx_meta);
s_if_meta = filtfilt(b_lp, a_lp, s_mix_meta);

fprintf('✓ 混频处理完成\n');

%% 6. 基础频域分析 (FFT)

% 6.1 空气介质差频 FFT
S_IF_air = fft(s_if_air, N);
S_IF_air_mag = abs(S_IF_air);

% 计算理论空气差频
f_beat_air_theory = K * tau_air;

% 6.2 超材料介质差频 FFT (加汉宁窗)
win = hann(N)';
s_if_meta_win = s_if_meta .* win;
S_IF_meta = fft(s_if_meta_win, N);
S_IF_meta_mag = abs(S_IF_meta) * 2; 

%% 7. 可视化 (Figure 1 - 8) - 与原脚本保持一致

% 辅助变量:限制频谱显示范围
f_range = [f_start-0.5e9, f_end+0.5e9];
f_indices = find(f >= f_range(1) & f <= f_range(2));

% --- Figure 1: 时域对比 ---
figure(1);
t_display = min(5e-6, T_m); 
idx_display = round(t_display/t_s);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, s_rx_air(1:idx_display), 'r--');
xlabel('时间 (μs)', 'FontName', 'SimHei'); 
ylabel('幅值', 'FontName', 'SimHei'); 
title('Figure 1: 发射信号 vs 空气接收', 'FontName', 'SimHei'); 
grid on;

% --- Figure 2: 频域对比 ---
figure(2);
S_TX_mag = abs(fft(s_tx));
S_RX_air_mag = abs(fft(s_rx_air));
plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_air_mag(f_indices), 'r--');
xlabel('频率 (GHz)', 'FontName', 'SimHei'); 
title('Figure 2: 发射 vs 空气接收 (频谱)', 'FontName', 'SimHei'); 
grid on;

% --- Figure 3: 时域对比 (超材料 - CST) ---
figure(3);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, real(s_rx_meta(1:idx_display)), 'r--');
xlabel('时间 (μs)', 'FontName', 'SimHei'); 
title('Figure 3: 发射信号 vs CST超材料接收', 'FontName', 'SimHei'); 
grid on;

% --- Figure 4: 频域对比 (超材料 - CST) ---
figure(4);
S_RX_meta_mag_plot = abs(S_RX_meta_fft);
plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_meta_mag_plot(f_indices), 'r--');
xlabel('频率 (GHz)', 'FontName', 'SimHei'); 
title('Figure 4: 发射 vs CST超材料接收 (频谱)', 'FontName', 'SimHei'); 
grid on;

% --- Figure 5-8: 差频信号显示 ---
% Figure 5: Air Time
figure(5);
t_if_disp = min(20e-6, T_m); idx_if = round(t_if_disp/t_s);
plot(t(1:idx_if)*1e6, s_if_air(1:idx_if), 'b');
title('Figure 5: 空气差频 (时域)', 'FontName', 'SimHei'); 
grid on;

% Figure 6: Air Freq
figure(6);
f_if_lim = 1e6; idx_if_f = round(f_if_lim/(f_s/N));
stem(f(1:idx_if_f)/1e3, S_IF_air_mag(1:idx_if_f), 'b', 'MarkerSize', 2);
xline(f_beat_air_theory/1e3, 'r--', 'LineWidth', 2);
title('Figure 6: 空气差频 (频谱)', 'FontName', 'SimHei'); 
grid on;

% Figure 7: Meta Time
figure(7);
plot(t(1:idx_if)*1e6, s_if_meta(1:idx_if), 'b');
title('Figure 7: CST超材料差频 (时域)', 'FontName', 'SimHei'); 
grid on;

% Figure 8: Meta Freq
figure(8);
stem(f(1:idx_if_f)/1e3, S_IF_meta_mag(1:idx_if_f), 'b', 'MarkerSize', 2);
title('Figure 8: CST超材料差频 (频谱)', 'FontName', 'SimHei'); 
grid on;

%% 8. 高级信号处理:滑动窗口 + MDL + ESPRIT + 幅度提取
fprintf('\n开始高级信号处理 (滑动窗口 + ESPRIT + 幅度加权)...\n');

% ---------------------------------------------------------
% 8.1 数据预处理
% ---------------------------------------------------------
decimation_factor = 200; 
f_s_proc = f_s / decimation_factor; 
s_proc = s_if_meta(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

% ---------------------------------------------------------
% 8.2 算法参数
% ---------------------------------------------------------
win_time = 12e-6;                
win_len = round(win_time * f_s_proc); 
step_len = round(win_len / 10);  
L_sub = round(win_len / 2);     

feature_f_probe = []; 
feature_tau_absolute = []; 
feature_amplitude = []; 

% ---------------------------------------------------------
% 8.3 处理循环
% ---------------------------------------------------------
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
    
    % --- 信号处理核心 (Hankel / FB / Eigen) ---
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
    
    % --- MDL 准则 ---
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
    
    % 鲁棒性限制
    num_sources = max(1, k_est); 
    num_sources = min(num_sources, 3);
    
    % --- TLS-ESPRIT ---
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    % 频率筛选
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6); 
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    [f_beat_est, best_idx_in_valid] = min(valid_freqs); 
    
    % --- 幅度提取 (用于加权) ---
    amp_est = rms(x_window); 
    
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];
    
    if mod(i, 50) == 0, waitbar(i/num_windows, hWait); end
end
close(hWait);

fprintf('✓ ESPRIT特征提取完成: %d 个有效点\n', length(feature_f_probe));

%% 9. 诊断结果可视化:CST数据与ESPRIT提取对比

% --- 9.1 从CST S21计算理论群时延 ---
phase_cst = angle(S21);  % 相位 (rad)
phase_unwrap = unwrap(phase_cst);  % 解缠绕
omega_cst = 2*pi*f_cst;

% 数值求导计算群时延
tau_cst = -diff(phase_unwrap) ./ diff(omega_cst);
tau_cst = [tau_cst; tau_cst(end)];  % 维度补齐

% 相对群时延 (减去真空时延)
tau_vacuum = d / c;
tau_cst_relative = tau_cst - tau_vacuum;

% --- 9.2 测量结果处理 ---
tau_relative_meas = feature_tau_absolute - tau_air;

% --- 9.3 绘图 ---
figure(9); clf;
set(gcf, 'Position', [100, 100, 900, 600]);

% 过滤异常值
valid_idx = tau_relative_meas > -5e-9 & tau_relative_meas < 10e-9;
scatter(feature_f_probe(valid_idx)/1e9, tau_relative_meas(valid_idx)*1e9, 20, 'b', 'filled', ...
    'DisplayName', '仿真测量点 (ESPRIT)');

hold on;

% 绘制CST理论曲线
plot(f_cst/1e9, tau_cst_relative*1e9, 'r', 'LineWidth', 2.5, ...
    'DisplayName', 'CST全波仿真理论值');

% 找到CST数据中的谐振频率 (S21幅度最小点)
[~, idx_res] = min(abs(S21));
f_res_cst = f_cst(idx_res);

% 标注谐振频率
xline(f_res_cst/1e9, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('谐振频率 %.2f GHz', f_res_cst/1e9));

grid on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相对群时延 Δτ (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title({['CST全波仿真超材料诊断结果'], ...
       ['CSRR-loaded WR-28 波导']}, ...
       'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 12);
xlim([f_start/1e9, f_end/1e9]);

fprintf('\n✓ Figure 9 绘图完成\n');

%% 10. 参数反演：Metropolis-Hastings MCMC 算法

fprintf('\n---------------------------------------------\n');
fprintf('开始参数反演 (MCMC - 基于CST数据)...\n');
fprintf('---------------------------------------------\n');

% -------------------------------------------------------------------------
% 10.1 数据筛选与准备
% -------------------------------------------------------------------------
tau_relative_meas = feature_tau_absolute - tau_air;

% 筛选数据
fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (abs(tau_relative_meas) < 10e-9); 

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);

if isempty(X_fit)
    error('有效拟合数据点为空,请检查 ESPRIT 提取结果!');
end

% 权重归一化
Weights = (W_raw / max(W_raw)).^2; 

% 测量噪声估计
sigma_meas = 0.1e-9;

% Lorentz模型参数 (用于反演)
omega_p_meta = 2*pi*5e9;  % 估计值

% -------------------------------------------------------------------------
% 10.2 MCMC 参数设置
% -------------------------------------------------------------------------
N_samples = 10000;
burn_in = 2000;

% 先验分布范围 (基于CST观察)
fres_min = 34e9; fres_max = 35.5e9;
gamma_min = 0.1e9; gamma_max = 3e9;

% 提议分布步长 (增大以降低接受率至20-50%)
sigma_fres = (fres_max - fres_min) * 0.08;   % 从0.015增至0.08
sigma_gamma = (gamma_max - gamma_min) * 0.15; % 从0.05增至0.15

% -------------------------------------------------------------------------
% 10.3 初始化
% -------------------------------------------------------------------------
rng(42);
fres_current = fres_min + (fres_max - fres_min) * rand();
gamma_current = gamma_min + (gamma_max - gamma_min) * rand();

fprintf('MCMC 初始点:\n');
fprintf('  f_res = %.3f GHz\n', fres_current/1e9);
fprintf('  gamma = %.3f GHz\n', gamma_current/1e9);

% 计算初始对数似然
logL_current = compute_log_likelihood_lorentz(X_fit, Y_fit, Weights, fres_current, gamma_current, ...
                                               sigma_meas, omega_p_meta, d, c);

% 存储采样结果
samples_fres = zeros(N_samples, 1);
samples_gamma = zeros(N_samples, 1);
samples_logL = zeros(N_samples, 1);
accept_count = 0;

% -------------------------------------------------------------------------
% 10.4 MCMC 主循环
% -------------------------------------------------------------------------
hWait = waitbar(0, 'MCMC 采样中...');

for i = 1:N_samples
    % 提议新参数
    fres_proposed = fres_current + sigma_fres * randn();
    gamma_proposed = gamma_current + sigma_gamma * randn();
    
    % 先验约束检查
    if fres_proposed < fres_min || fres_proposed > fres_max || ...
       gamma_proposed < gamma_min || gamma_proposed > gamma_max
        samples_fres(i) = fres_current;
        samples_gamma(i) = gamma_current;
        samples_logL(i) = logL_current;
        continue;
    end
    
    % 计算提议点的对数似然
    logL_proposed = compute_log_likelihood_lorentz(X_fit, Y_fit, Weights, fres_proposed, gamma_proposed, ...
                                                    sigma_meas, omega_p_meta, d, c);
    
    % Metropolis-Hastings 接受概率
    log_alpha = logL_proposed - logL_current;
    
    if log(rand()) < log_alpha
        fres_current = fres_proposed;
        gamma_current = gamma_proposed;
        logL_current = logL_proposed;
        accept_count = accept_count + 1;
    end
    
    samples_fres(i) = fres_current;
    samples_gamma(i) = gamma_current;
    samples_logL(i) = logL_current;
    
    if mod(i, 500) == 0
        waitbar(i/N_samples, hWait, sprintf('MCMC 采样中... %.0f%%', i/N_samples*100));
    end
end
close(hWait);

% -------------------------------------------------------------------------
% 10.5 后验分析
% -------------------------------------------------------------------------
fprintf('\n===== MCMC 采样完成 =====\n');
fprintf('总采样数: %d, 预烧期: %d, 有效样本: %d\n', N_samples, burn_in, N_samples - burn_in);
fprintf('接受率: %.2f%% (理想范围: 20-50%%)\n', accept_count/N_samples*100);

% 丢弃预烧期
samples_fres_valid = samples_fres(burn_in+1:end);
samples_gamma_valid = samples_gamma(burn_in+1:end);

% 后验统计
fres_mean = mean(samples_fres_valid);
fres_std = std(samples_fres_valid);
fres_ci = prctile(samples_fres_valid, [2.5, 97.5]);

gamma_mean = mean(samples_gamma_valid);
gamma_std = std(samples_gamma_valid);
gamma_ci = prctile(samples_gamma_valid, [2.5, 97.5]);

fprintf('\n--- 反演结果 ---\n');
fprintf('f_res:\n');
fprintf('  CST谐振频率: %.4f GHz\n', f_res_cst/1e9);
fprintf('  后验均值:    %.4f GHz\n', fres_mean/1e9);
fprintf('  后验标准差:  %.4f GHz\n', fres_std/1e9);
fprintf('  95%% CI:     [%.4f, %.4f] GHz\n', fres_ci(1)/1e9, fres_ci(2)/1e9);
fprintf('  相对误差:    %.2f%%\n', (fres_mean - f_res_cst)/f_res_cst * 100);

fprintf('\ngamma:\n');
fprintf('  后验均值:    %.4f GHz\n', gamma_mean/1e9);
fprintf('  后验标准差:  %.4f GHz\n', gamma_std/1e9);
fprintf('  95%% CI:     [%.4f, %.4f] GHz\n', gamma_ci(1)/1e9, gamma_ci(2)/1e9);

%% 11. MCMC可视化 (Figures 11-13)

% Figure 11: Trace plots
figure(11); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 600]);

subplot(2,2,1);
plot(samples_fres/1e9, 'b', 'LineWidth', 0.5);
hold on;
yline(f_res_cst/1e9, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数', 'FontName', 'SimHei'); ylabel('f_{res} (GHz)');
title('(a) f_{res} Trace Plot', 'FontName', 'SimHei'); grid on;
legend('采样链', 'CST参考值', 'Location', 'best');

subplot(2,2,2);
plot(samples_gamma/1e9, 'b', 'LineWidth', 0.5);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数', 'FontName', 'SimHei'); ylabel('\gamma (GHz)');
title('(b) \gamma Trace Plot', 'FontName', 'SimHei'); grid on;

subplot(2,2,3);
histogram(samples_fres_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
hold on;
xline(f_res_cst/1e9, 'r--', 'LineWidth', 2);
xline(fres_ci(1)/1e9, 'k--'); xline(fres_ci(2)/1e9, 'k--');
xlabel('f_{res} (GHz)'); ylabel('概率密度', 'FontName', 'SimHei');
title('(c) f_{res} 后验分布', 'FontName', 'SimHei'); grid on;

subplot(2,2,4);
histogram(samples_gamma_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
xline(gamma_ci(1)/1e9, 'k--'); xline(gamma_ci(2)/1e9, 'k--');
xlabel('\gamma (GHz)'); ylabel('概率密度', 'FontName', 'SimHei');
title('(d) \gamma 后验分布', 'FontName', 'SimHei'); grid on;

sgtitle('CST数据 Lorentz模型 MCMC反演结果', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

% Figure 12: Corner Plot
figure(12); clf;
set(gcf, 'Color', 'w', 'Position', [150 150 700 600]);

subplot(2,2,1);
histogram(samples_fres_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
xline(f_res_cst/1e9, 'r--', 'LineWidth', 2);
xlabel('f_{res} (GHz)'); ylabel('PDF');
title('f_{res} 边缘分布', 'FontName', 'SimHei');

subplot(2,2,4);
histogram(samples_gamma_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
xlabel('\gamma (GHz)'); ylabel('PDF');
title('\gamma 边缘分布', 'FontName', 'SimHei');

subplot(2,2,3);
scatter(samples_fres_valid(1:10:end)/1e9, samples_gamma_valid(1:10:end)/1e9, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(f_res_cst/1e9, gamma_mean/1e9, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('f_{res} (GHz)'); ylabel('\gamma (GHz)');
title('联合后验分布', 'FontName', 'SimHei');
grid on;

subplot(2,2,2);
corr_val = corrcoef(samples_fres_valid, samples_gamma_valid);
text(0.5, 0.5, sprintf('相关系数\n\\rho = %.3f', corr_val(1,2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 16);
axis off;
title('参数耦合分析', 'FontName', 'SimHei');

sgtitle('Corner Plot: f_{res} vs \gamma', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 13: 拟合验证
figure(13); clf;
set(gcf, 'Color', 'w', 'Position', [200 200 800 400]);

scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled'); 
colorbar; ylabel(colorbar, '权重');
hold on;

f_plot = linspace(min(X_fit), max(X_fit), 200);
tau_plot = calculate_lorentz_delay(f_plot, fres_mean, gamma_mean, omega_p_meta, d, c);
plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);

xline(fres_mean/1e9, 'g--', 'LineWidth', 2);

title(sprintf('MCMC拟合结果 (f_{res}误差: %.2f%%)', (fres_mean - f_res_cst)/f_res_cst * 100), 'FontName', 'SimHei'); 
xlabel('频率 (GHz)', 'FontName', 'SimHei'); ylabel('相对时延 (ns)', 'FontName', 'SimHei'); grid on;
legend('测量数据', '后验均值曲线', '谐振频率', 'Location', 'best');
xlim([min(X_fit)/1e9 max(X_fit)/1e9]);

fprintf('\n✓ 绘图完成: Figure 11 (Trace), Figure 12 (Corner), Figure 13 (Fit)\n');
fprintf('仿真完成!\n');

%% =========================================================================
%  局部函数
%  =========================================================================

function [f_Hz, S11, S21, S12, S22] = read_s2p_touchstone(filename)
    % 读取Touchstone .s2p文件
    % 返回频率(Hz)和复数S参数
    
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    % 读取头部信息
    freq_unit_factor = 1e9;  % 默认GHz
    format_type = 'MA';      % 默认 Magnitude-Angle
    
    data = [];
    while ~feof(fid)
        line = fgetl(fid);
        
        % 跳过空行
        if isempty(line), continue; end
        
        % 跳过注释行
        if line(1) == '!', continue; end
        
        % 解析选项行
        if line(1) == '#'
            tokens = strsplit(upper(line));
            % 解析频率单位
            if any(strcmp(tokens, 'HZ')), freq_unit_factor = 1; end
            if any(strcmp(tokens, 'KHZ')), freq_unit_factor = 1e3; end
            if any(strcmp(tokens, 'MHZ')), freq_unit_factor = 1e6; end
            if any(strcmp(tokens, 'GHZ')), freq_unit_factor = 1e9; end
            % 解析格式
            if any(strcmp(tokens, 'RI')), format_type = 'RI'; end
            if any(strcmp(tokens, 'MA')), format_type = 'MA'; end
            if any(strcmp(tokens, 'DB')), format_type = 'DB'; end
            continue;
        end
        
        % 数据行
        values = sscanf(line, '%f');
        if length(values) >= 9
            data = [data; values(1:9)'];
        end
    end
    fclose(fid);
    
    % 提取数据
    f_Hz = data(:, 1) * freq_unit_factor;
    
    % 根据格式转换为复数
    if strcmp(format_type, 'MA')
        % Magnitude-Angle (角度为度)
        S11 = data(:, 2) .* exp(1i * data(:, 3) * pi / 180);
        S21 = data(:, 4) .* exp(1i * data(:, 5) * pi / 180);
        S12 = data(:, 6) .* exp(1i * data(:, 7) * pi / 180);
        S22 = data(:, 8) .* exp(1i * data(:, 9) * pi / 180);
    elseif strcmp(format_type, 'RI')
        % Real-Imaginary
        S11 = data(:, 2) + 1i * data(:, 3);
        S21 = data(:, 4) + 1i * data(:, 5);
        S12 = data(:, 6) + 1i * data(:, 7);
        S22 = data(:, 8) + 1i * data(:, 9);
    elseif strcmp(format_type, 'DB')
        % dB-Angle
        S11 = 10.^(data(:, 2)/20) .* exp(1i * data(:, 3) * pi / 180);
        S21 = 10.^(data(:, 4)/20) .* exp(1i * data(:, 5) * pi / 180);
        S12 = 10.^(data(:, 6)/20) .* exp(1i * data(:, 7) * pi / 180);
        S22 = 10.^(data(:, 8)/20) .* exp(1i * data(:, 9) * pi / 180);
    end
end

function logL = compute_log_likelihood_lorentz(f_data, tau_data, weights, fres_val, gamma_val, sigma, wp_meta, d, c)
    % 计算 Lorentz 模型的加权对数似然函数
    
    % 物理约束检查
    if fres_val <= 0 || gamma_val <= 0
        logL = -1e10; return;
    end
    
    % 检查谐振频率是否在合理范围
    if fres_val < min(f_data)*0.8 || fres_val > max(f_data)*1.2
        logL = -1e10; return;
    end
    
    try
        % 计算理论时延
        tau_theory = calculate_lorentz_delay(f_data, fres_val, gamma_val, wp_meta, d, c);
        
        % 加权残差
        residuals = (tau_theory - tau_data) / sigma;
        
        % 对数似然 (高斯噪声模型)
        logL = -0.5 * sum(weights .* residuals.^2);
        
        % 检查 NaN
        if isnan(logL) || isinf(logL)
            logL = -1e10;
        end
    catch
        logL = -1e10;
    end
end

function tau_rel = calculate_lorentz_delay(f_vec, f_res_val, gamma_val, wp_meta, d, c)
    % Lorentz模型相位求导法
    % 计算相对群时延 = (超材料群时延) - (真空群时延)
    
    omega_vec = 2 * pi * f_vec(:)';
    omega_res_val = 2 * pi * f_res_val;
    
    % Lorentz模型复介电常数
    % ε(ω) = 1 + ωₚ²/(ω₀² - ω² - jγω)
    eps_r = 1 + (wp_meta^2) ./ (omega_res_val^2 - omega_vec.^2 - 1i*2*pi*gamma_val*omega_vec);
    
    % 复波数 k = (ω/c) * sqrt(ε_r)
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 超材料段的总相位 phi = -real(k) * d
    phi_meta = -real(k_vec) * d;
    
    % 数值微分求群时延 tau_g = -d(phi)/d(ω)
    d_phi = diff(phi_meta);
    d_omega = diff(omega_vec);
    
    tau_total = -d_phi ./ d_omega;
    
    % 维度补齐
    tau_total = [tau_total, tau_total(end)];
    
    % 减去真空时延
    tau_rel = tau_total - (d/c);
    tau_rel = tau_rel(:);
end
