%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW超材料诊断系统MATLAB仿真 - 基于Lorentz模型
% 功能:生成信号、Lorentz模型传播、混频、特征提取、参数反演
% 适用对象:开口谐振环(SRR)阵列或电磁超表面
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 仿真参数设置
clc; clear all; close all;

% LFMCW雷达参数
f_start = 34.2e9;            
f_end = 37.4e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

% 传播介质参数 (超材料)
tau_air = 4e-9;              % 空气总时延
tau_fs = 1.75e-9;            % 单侧自由空间时延
d = 150e-3;                  % 超材料厚度 (m)
f_res = 35.5e9;              % 谐振频率 (Hz) - 待反演参数
gamma = 0.5e9;               % 阻尼因子 (Hz) - 待反演参数
omega_p_meta = 2*pi*5e9;     % 等效等离子体频率 (用于Lorentz模型强度)

% 计算派生参数
c = 3e8;                     
epsilon_0 = 8.854e-12;
omega_res = 2*pi*f_res;      % 谐振角频率

t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% --- 构建正确的FFT频率轴 (包含负频率) ---
f = (0:N-1)*(f_s/N);         
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

fprintf('仿真参数设置完成\n');
fprintf('谐振频率 f_res: %.2f GHz\n', f_res/1e9);
fprintf('阻尼因子 gamma: %.2f GHz\n', gamma/1e9);

%% 2. LFMCW信号生成模块
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              
fprintf('LFMCW信号生成完成\n');

%% 3. 信号传播模拟模块 (Lorentz模型)

% 3.1 空气介质传播模拟
delay_samples_air = round(tau_air/t_s);
s_rx_air = [zeros(1, delay_samples_air) s_tx(1:end-delay_samples_air)];

% 3.2 超材料介质传播模拟
% 第一段:自由空间
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];

% 第二段:穿过超材料 (频域处理)
S_after_fs1 = fft(s_after_fs1);

% --- Lorentz模型复介电常数 ---
% epsilon_r = 1 + omega_p^2 / (omega_0^2 - omega^2 - j*gamma*omega)
% 注意:这里使用点除和点乘

% 防止除以零
omega_safe = omega; 
omega_safe(omega_safe == 0) = 1e-10; 

% 计算复介电常数 (Lorentz模型)
% ε(ω) = 1 + ωₚ²/(ω₀² - ω² - jγω)
denominator = omega_res^2 - omega_safe.^2 - 1i*gamma*omega_safe;
epsilon_r_complex = 1 + (omega_p_meta^2) ./ denominator;

% 处理直流分量 (omega=0)
epsilon_r_complex(omega == 0) = 1; 

% 复波数 k = w/c * sqrt(epsilon)
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);

% 强制物理衰减
k_real = real(k_complex); % 决定相位变化
k_imag = imag(k_complex); % 决定幅度衰减

% 传递函数 H = exp(-j*k_real*d - abs(k_imag)*d)
H_meta = exp(-1i * k_real * d - abs(k_imag) * d);

% 应用传递函数
S_after_meta = S_after_fs1 .* H_meta;
S_RX_meta_fft = S_after_meta; 

% 转回时域
s_after_meta = real(ifft(S_after_meta));

% 第三段:自由空间
s_rx_meta = [zeros(1, delay_samples_fs) s_after_meta(1:end-delay_samples_fs)];

fprintf('超材料传播模拟完成 (Lorentz模型)\n');

%% 4. 混频处理与差频信号提取

% 4.1 空气介质混频
s_mix_air = s_tx .* s_rx_air;

% 低通滤波器设计
fc_lp = 100e6;  
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));

s_if_air = filtfilt(b_lp, a_lp, s_mix_air);

% 4.2 超材料介质混频
s_mix_meta = s_tx .* real(s_rx_meta);
s_if_meta = filtfilt(b_lp, a_lp, s_mix_meta);

fprintf('混频处理完成\n');

%% 5. 基础频域分析 (FFT)

% 5.1 空气介质差频 FFT
S_IF_air = fft(s_if_air, N);
S_IF_air_mag = abs(S_IF_air);

% 计算理论空气差频
f_beat_air_theory = K * tau_air;

% 5.2 超材料介质差频 FFT (加汉宁窗)
win = hann(N)';
s_if_meta_win = s_if_meta .* win;
S_IF_meta = fft(s_if_meta_win, N);
S_IF_meta_mag = abs(S_IF_meta) * 2; 

%% 6. 可视化 (Figure 1 - 8)

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

% --- Figure 3: 时域对比 (超材料) ---
figure(3);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, real(s_rx_meta(1:idx_display)), 'r--');
xlabel('时间 (μs)', 'FontName', 'SimHei'); 
title('Figure 3: 发射信号 vs 超材料接收', 'FontName', 'SimHei'); 
grid on;

% --- Figure 4: 频域对比 (超材料) ---
figure(4);
S_RX_meta_mag_plot = abs(S_RX_meta_fft);
plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_meta_mag_plot(f_indices), 'r--');
xlabel('频率 (GHz)', 'FontName', 'SimHei'); 
title('Figure 4: 发射 vs 超材料接收 (频谱)', 'FontName', 'SimHei'); 
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
title('Figure 7: 超材料差频 (时域)', 'FontName', 'SimHei'); 
grid on;

% Figure 8: Meta Freq
figure(8);
stem(f(1:idx_if_f)/1e3, S_IF_meta_mag(1:idx_if_f), 'b', 'MarkerSize', 2);
title('Figure 8: 超材料差频 (频谱)', 'FontName', 'SimHei'); 
grid on;

%% 7. 高级信号处理:滑动窗口 + MDL + ESPRIT + 幅度提取
fprintf('开始高级信号处理 (滑动窗口 + ESPRIT + 幅度加权)...\n');

% ---------------------------------------------------------
% 7.1 数据预处理
% ---------------------------------------------------------
decimation_factor = 200; 
f_s_proc = f_s / decimation_factor; 
s_proc = s_if_meta(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc);

% ---------------------------------------------------------
% 7.2 算法参数
% ---------------------------------------------------------
win_time = 12e-6;                
win_len = round(win_time * f_s_proc); 
step_len = round(win_len / 10);  
L_sub = round(win_len / 2);     

feature_f_probe = []; 
feature_tau_absolute = []; 
feature_amplitude = []; 

% ---------------------------------------------------------
% 7.3 处理循环
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

%% 8. 诊断结果可视化:高精度物理对比

% --- 8.1 理论计算:基于Lorentz模型 (含阻尼) ---
f_theory_vec = linspace(f_start, f_end, 1000);
omega_theory = 2*pi*f_theory_vec;

% Lorentz模型复介电常数
epsilon_r_theory = 1 + (omega_p_meta^2) ./ (omega_res^2 - omega_theory.^2 - 1i*gamma*omega_theory);

% 复波数
k_theory = (omega_theory ./ c) .* sqrt(epsilon_r_theory);

% 相位变化
phase_meta = -real(k_theory) * d;

% 理论群时延计算 (数值求导法)
tau_total_meta_layer = -diff(phase_meta) ./ diff(omega_theory);
tau_total_meta_layer = [tau_total_meta_layer, tau_total_meta_layer(end)];

% 真空时延
tau_vacuum_layer = d / c;

% 最终理论相对时延
tau_relative_theory = tau_total_meta_layer - tau_vacuum_layer;

% --- 8.2 测量结果处理 ---
tau_relative_meas = feature_tau_absolute - tau_air;

% --- 8.3 绘图 ---
figure(9); clf;
set(gcf, 'Position', [100, 100, 900, 600]);

% 过滤异常值
valid_idx = tau_relative_meas > -5e-9 & tau_relative_meas < 10e-9;
scatter(feature_f_probe(valid_idx)/1e9, tau_relative_meas(valid_idx)*1e9, 20, 'b', 'filled', ...
    'DisplayName', '仿真测量点 (ESPRIT)');

hold on;

% 绘制理论曲线
plot(f_theory_vec/1e9, tau_relative_theory*1e9, 'r', 'LineWidth', 2.5, ...
    'DisplayName', 'Lorentz模型理论值 (含阻尼)');

% 标注谐振频率
xline(f_res/1e9, 'g--', 'LineWidth', 2, 'DisplayName', '谐振频率');

grid on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相对群时延 Δτ (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title({['超材料谐振特性诊断结果'], ...
       ['设定 f_{res} = ' num2str(f_res/1e9, '%.2f') ' GHz, γ = ' num2str(gamma/1e9, '%.2f') ' GHz']}, ...
       'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 12);
xlim([f_start/1e9, f_end/1e9]);

fprintf('绘图完成。\n');
fprintf('理论时延计算采用相位求导法:tau = -d(phi)/d(omega) - d/c\n');

%% 9. 参数反演：Metropolis-Hastings MCMC 算法 (替换 LM)

fprintf('---------------------------------------------\n');
fprintf('开始参数反演 (MCMC Metropolis-Hastings Algorithm - Lorentz模型)...\n');
fprintf('优势：无需初始值猜测，自动量化 γ 的不确定性\n');
fprintf('---------------------------------------------\n');

% -------------------------------------------------------------------------
% 9.1 数据筛选与准备
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

% -------------------------------------------------------------------------
% 9.2 MCMC 参数设置
% -------------------------------------------------------------------------
N_samples = 10000;
burn_in = 2000;

% 先验分布范围
% 参数1: f_res (谐振频率) - 范围 34-37 GHz
fres_min = 34e9; fres_max = 37e9;

% 参数2: gamma (阻尼因子) - 范围 0.1-2 GHz
gamma_min = 0.1e9; gamma_max = 2e9;

% 提议分布步长
sigma_fres = (fres_max - fres_min) * 0.015;  % 谐振频率步长
sigma_gamma = (gamma_max - gamma_min) * 0.05; % 阻尼步长 (不敏感，用更大步长)

% -------------------------------------------------------------------------
% 9.3 初始化 (从先验分布随机采样)
% -------------------------------------------------------------------------
rng(42);
fres_current = fres_min + (fres_max - fres_min) * rand();
gamma_current = gamma_min + (gamma_max - gamma_min) * rand();

fprintf('MCMC 初始点（随机采样自先验）:\n');
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
% 9.4 MCMC 主循环
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
% 9.5 后验分析
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

fprintf('\n--- 后验分布统计 ---\n');
fprintf('f_res:\n');
fprintf('  真值:     %.4f GHz\n', f_res/1e9);
fprintf('  后验均值: %.4f GHz\n', fres_mean/1e9);
fprintf('  后验标准差: %.4f GHz\n', fres_std/1e9);
fprintf('  95%% CI:   [%.4f, %.4f] GHz\n', fres_ci(1)/1e9, fres_ci(2)/1e9);
fprintf('  相对误差: %.2f%%\n', (fres_mean - f_res)/f_res * 100);

fprintf('\ngamma:\n');
fprintf('  真值:     %.4f GHz\n', gamma/1e9);
fprintf('  后验均值: %.4f GHz\n', gamma_mean/1e9);
fprintf('  后验标准差: %.4f GHz\n', gamma_std/1e9);
fprintf('  95%% CI:   [%.4f, %.4f] GHz\n', gamma_ci(1)/1e9, gamma_ci(2)/1e9);

% 关键结论
fprintf('\n===== 不确定性分析结论 =====\n');
cv_fres = fres_std / fres_mean;
cv_gamma = gamma_std / gamma_mean;
fprintf('f_res 变异系数 (CV): %.4f%% → 高精度可观测\n', cv_fres*100);
fprintf('gamma 变异系数 (CV): %.2f%% → ', cv_gamma*100);
if cv_gamma > 0.3
    fprintf('低精度，存在平底谷！建议固定此参数。\n');
else
    fprintf('可观测\n');
end

% -------------------------------------------------------------------------
% 9.6 可视化
% -------------------------------------------------------------------------

% Figure 11: Trace plots
figure(11); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 600]);

subplot(2,2,1);
plot(samples_fres/1e9, 'b', 'LineWidth', 0.5);
hold on;
yline(f_res/1e9, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数'); ylabel('f_{res} (GHz)');
title('(a) f_{res} Trace Plot'); grid on;
legend('采样链', '真值', 'Location', 'best');

subplot(2,2,2);
plot(samples_gamma/1e9, 'b', 'LineWidth', 0.5);
hold on;
yline(gamma/1e9, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数'); ylabel('\gamma (GHz)');
title('(b) \gamma Trace Plot'); grid on;

subplot(2,2,3);
histogram(samples_fres_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
hold on;
xline(f_res/1e9, 'r--', 'LineWidth', 2);
xline(fres_ci(1)/1e9, 'k--'); xline(fres_ci(2)/1e9, 'k--');
xlabel('f_{res} (GHz)'); ylabel('概率密度');
title('(c) f_{res} 后验分布'); grid on;

subplot(2,2,4);
histogram(samples_gamma_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
hold on;
xline(gamma/1e9, 'r--', 'LineWidth', 2);
xline(gamma_ci(1)/1e9, 'k--'); xline(gamma_ci(2)/1e9, 'k--');
xlabel('\gamma (GHz)'); ylabel('概率密度');
title('(d) \gamma 后验分布'); grid on;

sgtitle('Lorentz模型 MCMC 参数反演结果', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 12: Corner Plot
figure(12); clf;
set(gcf, 'Color', 'w', 'Position', [150 150 700 600]);

subplot(2,2,1);
histogram(samples_fres_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
xline(f_res/1e9, 'r--', 'LineWidth', 2);
xlabel('f_{res} (GHz)'); ylabel('PDF');
title('f_{res} 边缘分布');

subplot(2,2,4);
histogram(samples_gamma_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
xline(gamma/1e9, 'r--', 'LineWidth', 2);
xlabel('\gamma (GHz)'); ylabel('PDF');
title('\gamma 边缘分布');

subplot(2,2,3);
scatter(samples_fres_valid(1:10:end)/1e9, samples_gamma_valid(1:10:end)/1e9, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(f_res/1e9, gamma/1e9, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('f_{res} (GHz)'); ylabel('\gamma (GHz)');
title('联合后验分布 (Corner Plot)');
grid on;

subplot(2,2,2);
corr_val = corrcoef(samples_fres_valid, samples_gamma_valid);
text(0.5, 0.5, sprintf('相关系数\n\\rho = %.3f', corr_val(1,2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 16);
axis off;
title('参数耦合分析');

sgtitle('Lorentz模型 Corner Plot: f_{res} vs \gamma', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 13: 拟合验证
figure(13); clf;
set(gcf, 'Color', 'w', 'Position', [200 200 800 400]);

scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled'); 
colorbar; ylabel(colorbar, '权重');
hold on;

f_plot = linspace(min(X_fit), max(X_fit), 200);
tau_plot = calculate_lorentz_delay(f_plot, fres_mean, gamma_mean, omega_p_meta, d, c);
plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);

% 95% 置信带
n_curves = 100;
idx_sample = randperm(length(samples_fres_valid), min(n_curves, length(samples_fres_valid)));
for k = 1:length(idx_sample)
    tau_k = calculate_lorentz_delay(f_plot, samples_fres_valid(idx_sample(k)), ...
                                     samples_gamma_valid(idx_sample(k)), omega_p_meta, d, c);
    plot(f_plot/1e9, tau_k*1e9, 'Color', [0.8 0.8 0.8, 0.2], 'LineWidth', 0.5);
end

plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);
xline(fres_mean/1e9, 'g--', 'LineWidth', 2);

title(sprintf('MCMC拟合结果 (f_{res}误差: %.2f%%)', (fres_mean - f_res)/f_res * 100)); 
xlabel('频率 (GHz)'); ylabel('相对时延 (ns)'); grid on;
legend('测量数据', '后验均值曲线', '95%置信带', '谐振频率', 'Location', 'best');
xlim([min(X_fit)/1e9 max(X_fit)/1e9]);

fprintf('\n绘图完成: Figure 11 (Trace), Figure 12 (Corner), Figure 13 (Fit)\n');
fprintf('仿真完成!\n');

% =========================================================================
%  局部函数
% =========================================================================

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
    
    omega_vec = 2 * pi * f_vec;
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
end