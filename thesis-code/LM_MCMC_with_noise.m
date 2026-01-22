%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW等离子体诊断系统MATLAB仿真 - 含噪声版 (支持鲁棒性测试)
% 功能：生成信号、Drude模型传播、添加高斯白噪声、混频、MCMC反演
% 修订：在射频端口添加AWGN,真实模拟接收机链路噪声传递效应
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 仿真参数设置 (修正频率轴逻辑)
clc; clear all; close all;

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
f_c = 33e9;                  
nu = 1.5e9;                  

% 计算派生参数 (修正 n_e 定义防止报错)
c = 3e8;                     
omega_p = 2*pi*f_c;          
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;
n_e = (omega_p^2 * epsilon_0 * m_e) / e^2; % 显式计算 n_e

t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% --- 关键修正：构建正确的FFT频率轴 (包含负频率) ---
f = (0:N-1)*(f_s/N);         
% 将大于 fs/2 的频率映射为负频率
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

fprintf('仿真参数设置完成\n');
fprintf('电子密度 n_e: %.2e m^-3\n', n_e);

%% 2. LFMCW信号生成模块 (保持不变)
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              
fprintf('LFMCW信号生成完成\n');

%% 3. 信号传播模拟模块 (修正Drude模型计算)

% 3.1 空气介质传播模拟
delay_samples_air = round(tau_air/t_s);
s_rx_air = [zeros(1, delay_samples_air) s_tx(1:end-delay_samples_air)];

% 3.2 等离子体介质传播模拟 - 三段式
% 第一段：自由空间
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];

% 第二段：穿过等离子体 (频域处理)
S_after_fs1 = fft(s_after_fs1);

% --- 关键修正：向量化计算 H_plasma (处理正负频率) ---
% Drude模型复介电常数 (支持负频率omega输入)
% epsilon = 1 - wp^2 / (w*(w + i*nu))
% 注意：这里使用点除和点乘

% 1. 防止除以零
omega_safe = omega; 
omega_safe(omega_safe == 0) = 1e-10; 

% 2. 计算复介电常数
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_complex(omega == 0) = 1; 

% 3. 计算复波数 k
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);

% 4. 计算传递函数 (强制让虚部变成衰减)
k_real = real(k_complex); % 决定相位变化
k_imag = imag(k_complex); % 决定幅度衰减

% 关键修正：exp(-abs(k_imag)*d) 确保信号一定变弱，不会爆炸
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

% 应用传递函数
S_after_plasma = S_after_fs1 .* H_plasma;
S_RX_plasma_fft = S_after_plasma; 

% 转回时域 (取real防止微小虚部残留)
s_after_plasma = real(ifft(S_after_plasma));

% 第三段：自由空间
s_rx_plasma = [zeros(1, delay_samples_fs) s_after_plasma(1:end-delay_samples_fs)];

fprintf('等离子体介质传播模拟完成 (已修正FFT频率轴)\n');

% =========================================================================
% 【核心修正】加入加性高斯白噪声 (AWGN) 以模拟真实测量环境
% =========================================================================
SNR_dB = 20; % 设置信噪比 (dB)，典型值取 20dB

% 计算信号功率
Ps = mean(s_rx_plasma.^2);

% 根据 SNR 计算噪声功率: SNR = 10*log10(Ps/Pn) => Pn = Ps / 10^(SNR/10)
Pn = Ps / (10^(SNR_dB/10));

% 生成高斯白噪声 (标准差为 sqrt(Pn))
noise = sqrt(Pn) * randn(size(s_rx_plasma));

% 叠加噪声
s_rx_plasma_pure = s_rx_plasma; % 保留一份纯净信号备用(可选)
s_rx_plasma = s_rx_plasma + noise;

fprintf('已加入高斯白噪声: SNR = %d dB\n', SNR_dB);
fprintf('  信号功率 Ps = %.2e W\n', Ps);
fprintf('  噪声功率 Pn = %.2e W\n', Pn);
fprintf('  实际SNR = %.2f dB\n', 10*log10(Ps/Pn));
% =========================================================================

%% 4. 混频处理与差频信号提取

% 4.1 空气介质混频
s_mix_air = s_tx .* s_rx_air;

% 低通滤波器设计 (提取差频)
fc_lp = 100e6;  % 截止频率100MHz (足以覆盖差频)
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));

s_if_air = filtfilt(b_lp, a_lp, s_mix_air);

% 4.2 等离子体介质混频 (含噪声信号)
s_mix_plasma = s_tx .* real(s_rx_plasma);
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

fprintf('混频处理与差频信号提取完成 (已混频噪声)\n');

%% 5. 基础频域分析 (FFT) - 仅用于绘图数据准备

% 5.1 空气介质差频 FFT
S_IF_air = fft(s_if_air, N);
S_IF_air_mag = abs(S_IF_air);

% 计算理论空气差频 (仅画图参考用)
f_beat_air_theory = K * tau_air;

% 5.2 等离子体介质差频 FFT (加汉宁窗抑制旁瓣)
win = hann(N)';
s_if_plasma_win = s_if_plasma .* win;
S_IF_plasma = fft(s_if_plasma_win, N);
S_IF_plasma_mag = abs(S_IF_plasma) * 2; % 补偿窗函数幅度损失

%% 6. 可视化 (Figure 1 - 8)

% 辅助变量：限制频谱显示范围
f_range = [f_start-0.5e9, f_end+0.5e9];
f_indices = find(f >= f_range(1) & f <= f_range(2));

% --- Figure 1: 时域对比 ---
figure(1);
t_display = min(5e-6, T_m); 
idx_display = round(t_display/t_s);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, s_rx_air(1:idx_display), 'r--');
xlabel('时间 (μs)'); ylabel('幅值'); title('Figure 1: 发射信号 vs 空气接收'); grid on;

% --- Figure 2: 频域对比 (修正了报错) ---
figure(2);
S_TX_mag = abs(fft(s_tx));
S_RX_air_mag = abs(fft(s_rx_air));
% 直接画有效范围，不画全谱，避免报错
plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_air_mag(f_indices), 'r--');
xlabel('频率 (GHz)'); title('Figure 2: 发射 vs 空气接收 (频谱)'); grid on;

% --- Figure 3: 时域对比 (等离子体 - 含噪声) ---
figure(3);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, real(s_rx_plasma(1:idx_display)), 'r--');
xlabel('时间 (μs)'); title('Figure 3: 发射信号 vs 等离子体接收 (含噪)'); grid on;

% --- Figure 4: 频域对比 (等离子体 - 含噪声) ---
figure(4);
S_RX_plasma_mag_plot = abs(S_RX_plasma_fft);
plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_plasma_mag_plot(f_indices), 'r--');
xlabel('频率 (GHz)'); title('Figure 4: 发射 vs 等离子体接收 (频谱,含噪)'); grid on;

% --- Figure 5-8: 差频信号显示 ---
% Figure 5: Air Time
figure(5);
t_if_disp = min(20e-6, T_m); idx_if = round(t_if_disp/t_s);
plot(t(1:idx_if)*1e6, s_if_air(1:idx_if), 'b');
title('Figure 5: 空气差频 (时域)'); grid on;

% Figure 6: Air Freq
figure(6);
f_if_lim = 1e6; idx_if_f = round(f_if_lim/(f_s/N));
stem(f(1:idx_if_f)/1e3, S_IF_air_mag(1:idx_if_f), 'b', 'MarkerSize', 2);
xline(f_beat_air_theory/1e3, 'r--', 'LineWidth', 2);
title('Figure 6: 空气差频 (频谱)'); grid on;

% Figure 7: Plasma Time (含噪声)
figure(7);
plot(t(1:idx_if)*1e6, s_if_plasma(1:idx_if), 'b');
title('Figure 7: 等离子体差频 (时域,含噪)'); grid on;

% Figure 8: Plasma Freq (含噪声)
figure(8);
stem(f(1:idx_if_f)/1e3, S_IF_plasma_mag(1:idx_if_f), 'b', 'MarkerSize', 2);
title('Figure 8: 等离子体差频 (频谱,含噪)'); grid on;

% --- 新增 Figure 9: SNR 可视化对比 ---
figure(10);
plot(t(1:idx_display)*1e6, s_rx_plasma_pure(1:idx_display), 'g', 'LineWidth', 1.5);
hold on;
plot(t(1:idx_display)*1e6, s_rx_plasma(1:idx_display), 'b', 'LineWidth', 0.8);
legend('纯净信号', sprintf('含噪信号 (SNR=%ddB)', SNR_dB));
xlabel('时间 (μs)'); ylabel('幅值'); 
title('Figure 10: 噪声影响对比'); grid on;

%% 7. 高级信号处理：滑动窗口 + MDL + ESPRIT + 【幅度提取】
fprintf('开始高级信号处理 (滑动窗口 + ESPRIT + 幅度加权)...\n');

% ---------------------------------------------------------
% 7.1 数据预处理
% ---------------------------------------------------------
decimation_factor = 200; 
f_s_proc = f_s / decimation_factor; 
s_proc = s_if_plasma(1:decimation_factor:end);
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
feature_amplitude = []; % <--- 新增：存储信号幅度作为权重

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
    num_sources = min(num_sources, 3); % 限制最大信源数，防止过拟合
    
    % --- TLS-ESPRIT ---
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    % 频率筛选 (直达波假设：最小频率)
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6); 
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs), continue; end
    
    [f_beat_est, best_idx_in_valid] = min(valid_freqs); 
    
    % --- 新增：幅度提取 (用于加权) ---
    % 利用最小二乘反解幅度: s(n) = A * exp(j*w*n)
    % 这里简单用最大特征值对应能量近似，或者用 x_window 的能量
    % 更精确的方法是解线性方程，但对于加权，用窗口平均能量足矣
    amp_est = rms(x_window); 
    
    tau_est = f_beat_est / K;
    
    feature_f_probe = [feature_f_probe, f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est]; % 存储权重
    
    if mod(i, 50) == 0, waitbar(i/num_windows, hWait); end
end
close(hWait);

%% 8. 诊断结果可视化：高精度物理对比

% --- 8.0 补充缺失的物理常量定义 ---
% 防止 workspace 清理后 n_e 丢失
e = 1.602e-19; 
m_e = 9.109e-31; 
epsilon_0 = 8.854e-12; 
c = 3e8;
% 根据截止频率反推电子密度
n_e = (2*pi*f_c)^2 * epsilon_0 * m_e / e^2; 

% --- 8.1 理论计算：基于全复数 Drude 模型 (含碰撞频率) ---
% 你的理解是完全正确的：不能忽略碰撞频率 nu，也不能简单用实部近似 vg。
% 最准确的方法是计算传输函数 H 的相位，然后对角频率求导：tau_g = -d(phi)/d(omega)

f_theory_vec = linspace(f_start, f_end, 1000); %以此频率轴计算理论值
omega_theory = 2*pi*f_theory_vec;

% 1. 复介电常数 (含碰撞项)
% epsilon_r = 1 - (omega_p^2) ./ (omega * (omega - j*nu))
% 注意：MATLAB中 1i 代表虚数单位
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_theory .* (omega_theory + 1i*nu));

% 2. 复波数 k = omega/c * sqrt(epsilon_r)
k_complex = (omega_theory ./ c) .* sqrt(epsilon_r_complex);

% 3. 等离子体段的相位变化 phi = -real(k) * d
% 负号是因为波函数通常定义为 exp(j(wt - kz))
phase_plasma = -real(k_complex) * d;

% 4. 理论相对时延计算 (数值求导法)
% 群时延 tau_g_total = - d(phi) / d(omega)
% 我们需要的是相对时延： (等离子体总时延) - (等离子体厚度对应的真空时延)
% 等离子体总时延(含d厚度):
tau_total_plasma_layer = -diff(phase_plasma) ./ diff(omega_theory);
% 补齐长度(diff少一个点)
tau_total_plasma_layer = [tau_total_plasma_layer, tau_total_plasma_layer(end)];

% 真空穿过同样厚度d的时延
tau_vacuum_layer = d / c;

% 最终理论相对时延
tau_relative_theory = tau_total_plasma_layer - tau_vacuum_layer;

% --- 8.2 测量结果处理 ---
% 测量值 = (自由空间1 + 等离子体绝对时延 + 自由空间2)
% 相对值 = 测量值 - 空气仿真总时延(tau_air)
tau_relative_meas = feature_tau_absolute - tau_air;

% --- 8.3 绘图 ---
figure(11); clf;
set(gcf, 'Position', [100, 100, 900, 600]);

% 绘制测量点 (蓝色散点)
% 过滤掉异常的负值或极小值(由ESPRIT在信号边缘失效引起)
valid_idx = tau_relative_meas > 0 & tau_relative_meas < 5e-9;
scatter(feature_f_probe(valid_idx)/1e9, tau_relative_meas(valid_idx)*1e9, 20, 'b', 'filled', ...
    'DisplayName', sprintf('仿真测量点 (SNR=%ddB)', SNR_dB));

hold on;

% 绘制修正后的理论曲线 (红色实线)
plot(f_theory_vec/1e9, tau_relative_theory*1e9, 'r', 'LineWidth', 2.5, ...
    'DisplayName', '全复数Drude模型理论值 (含碰撞)');

grid on;
xlabel('探测频率 (GHz)', 'FontSize', 12);
ylabel('相对群时延 \Delta\tau (ns)', 'FontSize', 12);
title({['等离子体电子密度诊断结果 (含噪声环境)'], ...
       ['设定 N_e = ' num2str(n_e, '%.2e') ' m^{-3}, \nu_e = ' num2str(nu/1e9) ' GHz, SNR = ' num2str(SNR_dB) ' dB']}, ...
       'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 12);
xlim([f_start/1e9, f_end/1e9]);

% 自动调整Y轴范围以聚焦数据
y_max = max(tau_relative_theory*1e9) * 1.2;
ylim([0, y_max]);

fprintf('绘图完成。\n');
fprintf('理论时延计算采用相位求导法：tau = -d(phi)/d(omega) - d/c\n');



%% 9. 参数反演：Metropolis-Hastings MCMC 算法 (替换 LM)
fprintf('---------------------------------------------\n');
fprintf('开始参数反演 (MCMC Metropolis-Hastings Algorithm)...\n');
fprintf('优势：无需初始值猜测，自动量化不确定性\n');
fprintf('---------------------------------------------\n');

% -------------------------------------------------------------------------
% 9.1 数据筛选与准备
% -------------------------------------------------------------------------
tau_relative_meas = feature_tau_absolute - tau_air;

% 筛选数据 (物理约束 + 去除边缘异常点)
fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (tau_relative_meas > 1e-11);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);

if isempty(X_fit)
    error('有效拟合数据点为空，请检查 ESPRIT 提取结果！');
end

% 权重归一化
Weights = (W_raw / max(W_raw)).^2; 

% 测量噪声估计 (用于似然函数) - 根据SNR动态调整
% 在高SNR时，测量误差主要来自算法,取值可稍小
% 在低SNR时,噪声主导,应该放大 sigma_meas
sigma_meas = 0.1e-9 * (1 + exp(-(SNR_dB-10)/5)); % 自适应噪声估计

% -------------------------------------------------------------------------
% 9.2 MCMC 参数设置
% -------------------------------------------------------------------------
N_samples = 10000;      % 总采样次数
burn_in = 2000;         % 预烧期

% 先验分布范围 (物理约束)
% 参数1: n_e (电子密度) - 范围 1e18 ~ 1e20 m^-3
ne_min = 1e18; ne_max = 1e20;

% 参数2: nu_e (碰撞频率) - 范围 0.1 ~ 5 GHz
nu_min = 0.1e9; nu_max = 5e9;

% 提议分布步长 (高斯随机游走)
% 经验法则：约为参数范围的 1-5%
sigma_ne = (ne_max - ne_min) * 0.02;  % n_e 步长
sigma_nu = (nu_max - nu_min) * 0.05;  % nu 步长 (不敏感参数用更大步长)

% -------------------------------------------------------------------------
% 9.3 初始化 (从先验分布随机采样，无需人工猜测！)
% -------------------------------------------------------------------------
rng(42); % 可复现性
ne_current = ne_min + (ne_max - ne_min) * rand(); % 均匀随机
nu_current = nu_min + (nu_max - nu_min) * rand();

fprintf('MCMC 初始点（随机采样自先验）:\n');
fprintf('  n_e = %.3e m^-3\n', ne_current);
fprintf('  nu_e = %.2f GHz\n', nu_current/1e9);

% 计算初始对数似然
logL_current = compute_log_likelihood(X_fit, Y_fit, Weights, ne_current, nu_current, ...
                                       sigma_meas, d, c, epsilon_0, m_e, e);

% 存储采样结果
samples_ne = zeros(N_samples, 1);
samples_nu = zeros(N_samples, 1);
samples_logL = zeros(N_samples, 1);
accept_count = 0;

% -------------------------------------------------------------------------
% 9.4 MCMC 主循环
% -------------------------------------------------------------------------
hWait = waitbar(0, 'MCMC 采样中...');

for i = 1:N_samples
    % 提议新参数 (对称高斯随机游走)
    ne_proposed = ne_current + sigma_ne * randn();
    nu_proposed = nu_current + sigma_nu * randn();
    
    % 先验约束检查 (超出范围直接拒绝)
    if ne_proposed < ne_min || ne_proposed > ne_max || ...
       nu_proposed < nu_min || nu_proposed > nu_max
        % 拒绝，保持当前状态
        samples_ne(i) = ne_current;
        samples_nu(i) = nu_current;
        samples_logL(i) = logL_current;
        continue;
    end
    
    % 计算提议点的对数似然
    logL_proposed = compute_log_likelihood(X_fit, Y_fit, Weights, ne_proposed, nu_proposed, ...
                                            sigma_meas, d, c, epsilon_0, m_e, e);
    
    % Metropolis-Hastings 接受概率 (对数形式避免下溢)
    log_alpha = logL_proposed - logL_current;
    
    % 接受或拒绝
    if log(rand()) < log_alpha
        % 接受
        ne_current = ne_proposed;
        nu_current = nu_proposed;
        logL_current = logL_proposed;
        accept_count = accept_count + 1;
    end
    
    % 存储当前状态
    samples_ne(i) = ne_current;
    samples_nu(i) = nu_current;
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
samples_ne_valid = samples_ne(burn_in+1:end);
samples_nu_valid = samples_nu(burn_in+1:end);

% 后验统计
ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]); % 95% 置信区间

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);

fprintf('\n--- 后验分布统计 ---\n');
fprintf('n_e:\n');
fprintf('  真值:     %.4e m^-3\n', n_e);
fprintf('  后验均值: %.4e m^-3\n', ne_mean);
fprintf('  后验标准差: %.2e m^-3\n', ne_std);
fprintf('  95%% CI:   [%.2e, %.2e]\n', ne_ci(1), ne_ci(2));
fprintf('  相对误差: %.2f%%\n', (ne_mean - n_e)/n_e * 100);

fprintf('\nnu_e:\n');
fprintf('  真值:     %.2f GHz\n', nu/1e9);
fprintf('  后验均值: %.2f GHz\n', nu_mean/1e9);
fprintf('  后验标准差: %.2f GHz\n', nu_std/1e9);
fprintf('  95%% CI:   [%.2f, %.2f] GHz\n', nu_ci(1)/1e9, nu_ci(2)/1e9);

% 关键结论
fprintf('\n===== 不确定性分析结论 =====\n');
cv_ne = ne_std / ne_mean;  % 变异系数
cv_nu = nu_std / nu_mean;
fprintf('n_e 变异系数 (CV): %.2f%% → 高精度可观测\n', cv_ne*100);
fprintf('nu_e 变异系数 (CV): %.2f%% → ', cv_nu*100);
if cv_nu > 0.5
    fprintf('低精度，存在平底谷！建议固定此参数。\n');
else
    fprintf('可观测\n');
end

% -------------------------------------------------------------------------
% 9.6 可视化
% -------------------------------------------------------------------------

% Figure 12: Trace plots
figure(12); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 600]);

subplot(2,2,1);
plot(samples_ne, 'b', 'LineWidth', 0.5);
hold on;
yline(n_e, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数'); ylabel('n_e (m^{-3})');
title(sprintf('(a) n_e Trace Plot (SNR=%ddB)', SNR_dB)); grid on;
legend('采样链', '真值', 'Location', 'best');

subplot(2,2,2);
plot(samples_nu/1e9, 'b', 'LineWidth', 0.5);
hold on;
yline(nu/1e9, 'r--', 'LineWidth', 2);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数'); ylabel('\nu_e (GHz)');
title('(b) \nu_e Trace Plot'); grid on;

subplot(2,2,3);
histogram(samples_ne_valid, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
hold on;
xline(n_e, 'r--', 'LineWidth', 2);
xline(ne_ci(1), 'k--'); xline(ne_ci(2), 'k--');
xlabel('n_e (m^{-3})'); ylabel('概率密度');
title('(c) n_e 后验分布'); grid on;

subplot(2,2,4);
histogram(samples_nu_valid/1e9, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
hold on;
xline(nu/1e9, 'r--', 'LineWidth', 2);
xline(nu_ci(1)/1e9, 'k--'); xline(nu_ci(2)/1e9, 'k--');
xlabel('\nu_e (GHz)'); ylabel('概率密度');
title('(d) \nu_e 后验分布'); grid on;

sgtitle(sprintf('Drude模型 MCMC 参数反演结果 (SNR=%ddB)', SNR_dB), 'FontSize', 14, 'FontWeight', 'bold');

% Figure 13: Corner Plot (2D 联合分布)
figure(13); clf;
set(gcf, 'Color', 'w', 'Position', [150 150 700 600]);

% 主对角线: 边缘分布
subplot(2,2,1);
histogram(samples_ne_valid, 40, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
xline(n_e, 'r--', 'LineWidth', 2);
xlabel('n_e (m^{-3})'); ylabel('PDF');
title('n_e 边缘分布');

subplot(2,2,4);
histogram(samples_nu_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
xline(nu/1e9, 'r--', 'LineWidth', 2);
xlabel('\nu_e (GHz)'); ylabel('PDF');
title('\nu_e 边缘分布');

% 下三角: 联合分布
subplot(2,2,3);
scatter(samples_ne_valid(1:10:end), samples_nu_valid(1:10:end)/1e9, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(n_e, nu/1e9, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('n_e (m^{-3})'); ylabel('\nu_e (GHz)');
title('联合后验分布 (Corner Plot)');
grid on;

% 上三角: 相关系数
subplot(2,2,2);
corr_val = corrcoef(samples_ne_valid, samples_nu_valid);
text(0.5, 0.5, sprintf('相关系数\\newline\\rho = %.3f', corr_val(1,2)), ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'Interpreter', 'tex');
axis off;
title('参数耦合分析');

sgtitle(sprintf('Drude模型 Corner Plot: n_e vs \\nu_e (SNR=%ddB)', SNR_dB), 'FontSize', 14, 'FontWeight', 'bold');

% Figure 14: 拟合验证
figure(14); clf;
set(gcf, 'Color', 'w', 'Position', [200 200 800 400]);

scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled'); 
colorbar; ylabel(colorbar, '权重');
hold on;

f_plot = linspace(min(X_fit), max(X_fit), 200);
tau_plot = calculate_theoretical_delay(f_plot, ne_mean, nu_mean, d, c, epsilon_0, m_e, e);
plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);

% 95% 置信带 (采样多条曲线)
n_curves = 100;
idx_sample = randperm(length(samples_ne_valid), min(n_curves, length(samples_ne_valid)));
for k = 1:length(idx_sample)
    tau_k = calculate_theoretical_delay(f_plot, samples_ne_valid(idx_sample(k)), ...
                                         samples_nu_valid(idx_sample(k)), d, c, epsilon_0, m_e, e);
    plot(f_plot/1e9, tau_k*1e9, 'Color', [0.8 0.8 0.8, 0.2], 'LineWidth', 0.5);
end

% 重新绘制均值曲线在最上层
plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);

title(sprintf('MCMC拟合结果 (n_e误差: %.2f%%, SNR=%ddB)', (ne_mean - n_e)/n_e * 100, SNR_dB)); 
xlabel('频率 (GHz)'); ylabel('相对时延 (ns)'); grid on;
legend('测量数据', '后验均值曲线', '95%置信带 (灰)', 'Location', 'best');
xlim([min(X_fit)/1e9 max(X_fit)/1e9]);

fprintf('\n绘图完成: Figure 11 (Trace), Figure 12 (Corner), Figure 13 (Fit)\n');

% =========================================================================
%  局部函数
% =========================================================================

function logL = compute_log_likelihood(f_data, tau_data, weights, ne_val, nu_val, sigma, d, c, eps0, me, e_charge)
    % 计算加权对数似然函数
    % logL = -0.5 * sum( w_i * ((y_i - f(theta))^2 / sigma^2) )
    
    % 物理约束检查
    if ne_val <= 0
        logL = -1e10; return;
    end
    
    % 计算截止频率
    fc_val = sqrt(ne_val * e_charge^2 / (eps0 * me)) / (2*pi);
    
    % 如果截止频率进入探测区，返回极小似然
    if fc_val >= (min(f_data) - 0.05e9)
        logL = -1e10; return;
    end
    
    try
        % 计算理论时延
        tau_theory = calculate_theoretical_delay(f_data, ne_val, nu_val, d, c, eps0, me, e_charge);
        
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

function tau_rel = calculate_theoretical_delay(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
    % 核心物理模型：Drude模型相位求导法
    % 计算相对群时延 = (等离子体群时延) - (真空群时延)
    
    omega_vec = 2 * pi * f_vec;
    wp_val = sqrt(ne_val * e_charge^2 / (eps0 * me));
    
    % Drude 模型复介电常数 (含碰撞频率虚部)
    % epsilon = 1 - wp^2 / (w*(w + i*nu))
    eps_r = 1 - (wp_val^2) ./ (omega_vec .* (omega_vec + 1i*nu_val));
    
    % 复波数 k = (w/c) * sqrt(eps_r)
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 等离子体段的总相位 phi = -real(k) * d
    phi_plasma = -real(k_vec) * d;
    
    % 数值微分求群时延 tau_g = -d(phi)/d(omega)
    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    
    tau_total = -d_phi ./ d_omega;
    
    % 维度补齐 (diff会少一个点，这里简单复制最后一个值)
    tau_total = [tau_total, tau_total(end)];
    
    % 减去真空穿过同样厚度 d 的时延 d/c
    % 得到的就是 "等离子体引起的附加时延"
    tau_rel = tau_total - (d/c);
end
