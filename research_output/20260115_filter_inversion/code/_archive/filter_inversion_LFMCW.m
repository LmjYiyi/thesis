%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW滤波器诊断系统MATLAB仿真 - 优化版（降低计算量）
% 参考代码：thesis-code/LM.m
% 创建日期：2026-01-15
% 优化策略：降低采样频率、缩短调制周期、简化信号处理
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 仿真参数设置
clc; clear all; close all;

% =========================================================================
% LFMCW雷达参数（优化后）
% 参考LM.m的参数设置，适当调整
% =========================================================================
f_start = 10e9;              % 扫频起始频率 (Hz) - 滤波器通带下限
f_end = 18e9;                % 扫频终止频率 (Hz) - 滤波器通带上限
T_m = 50e-6;                 % 调制周期 (s) - 与LM.m一致
B = f_end - f_start;         % 带宽 8 GHz
K = B/T_m;                   % 调频斜率 (Hz/s)
f_s = 50e9;                  % 采样频率 (Hz) - 降低到50GHz

% =========================================================================
% 滤波器真实参数（待反演）
% =========================================================================
F0_true = 14e9;              % 真实中心频率 (Hz)
BW_true = 8e9;               % 真实带宽 (Hz)
N_true = 5;                  % 真实滤波器阶数

% =========================================================================
% 传播参数
% =========================================================================
tau_ref = 2e-9;              % 参考时延（电缆等）

% =========================================================================
% 派生参数
% =========================================================================
t_s = 1/f_s;                 % 采样间隔
N_samples = round(T_m/t_s);  % 总采样点数
t = (0:N_samples-1)*t_s;     % 时间轴

% 构建FFT频率轴
f = (0:N_samples-1)*(f_s/N_samples);
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;

fprintf('===== LFMCW滤波器诊断系统仿真（优化版）=====\n');
fprintf('采样点数: %d (%.1f M点)\n', N_samples, N_samples/1e6);
fprintf('LFMCW参数: %.1f - %.1f GHz, T_m = %.0f μs\n', f_start/1e9, f_end/1e9, T_m*1e6);
fprintf('滤波器参数: F0 = %.2f GHz, BW = %.2f GHz, N = %d\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('=============================================\n\n');

%% 2. LFMCW信号生成
f_t = f_start + K*mod(t, T_m);   % 瞬时频率
phi_t = 2*pi*cumsum(f_t)*t_s;    % 瞬时相位
s_tx = cos(phi_t);               % 发射信号

fprintf('LFMCW信号生成完成\n');

%% 3. 信号传播模拟：通过滤波器
fprintf('开始滤波器传播模拟...\n');

% 3.1 参考通道（仅延迟）
delay_samples_ref = round(tau_ref/t_s);
s_rx_ref = [zeros(1, delay_samples_ref) s_tx(1:end-delay_samples_ref)];

% 3.2 滤波器通道（频域处理）
S_tx = fft(s_tx);

% 构建滤波器传递函数
H_filter = zeros(size(f));
for idx = 1:length(f)
    f_curr = abs(f(idx));
    if f_curr > 0
        % 幅度响应（Butterworth带通）
        x = (f_curr - F0_true) / (BW_true/2);
        H_mag = 1 / sqrt(1 + x^(2*N_true));
        
        % 群时延
        tau_g = (2*N_true) / (pi*BW_true) * (1 + x^2)^(-(N_true+1)/2);
        
        % 相位响应 phi = -omega * tau_g（简化模型）
        omega_curr = 2*pi*f_curr;
        phi = -omega_curr * tau_g;
        
        H_filter(idx) = H_mag * exp(1i * phi);
    end
end

% 参考时延
H_ref = exp(-1i * 2*pi*f * tau_ref);

% 组合：滤波器 + 参考时延
H_total = H_filter .* H_ref;

% 应用传递函数
S_rx = S_tx .* H_total;
s_rx_filter = real(ifft(S_rx));

fprintf('滤波器传播模拟完成\n');

%% 4. 混频处理
% 滤波器通道混频
s_mix_filter = s_tx .* s_rx_filter;

% 低通滤波
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_filter = filtfilt(b_lp, a_lp, s_mix_filter);

% 参考通道混频
s_mix_ref = s_tx .* s_rx_ref;
s_if_ref = filtfilt(b_lp, a_lp, s_mix_ref);

fprintf('混频处理完成\n');

%% 5. 基础可视化

figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);

subplot(1,2,1);
t_display = 5e-6;
idx_display = round(t_display/t_s);
plot(t(1:idx_display)*1e6, s_if_ref(1:idx_display), 'b', ...
     t(1:idx_display)*1e6, s_if_filter(1:idx_display), 'r');
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('差频信号对比', 'FontName', 'SimHei');
legend('参考通道', '滤波器通道');
grid on;

subplot(1,2,2);
S_IF = abs(fft(s_if_filter));
f_if = (0:N_samples-1)*(f_s/N_samples);
f_if_lim = 5e6;
idx_lim = round(f_if_lim/(f_s/N_samples));
plot(f_if(1:idx_lim)/1e3, S_IF(1:idx_lim));
xlabel('频率 (kHz)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('差频信号频谱', 'FontName', 'SimHei');
grid on;

%% 6. 滑动窗口 + ESPRIT 特征提取
fprintf('\n开始ESPRIT特征提取...\n');

% 降采样（对参考通道和滤波器通道都处理）
decimation_factor = 100;
f_s_proc = f_s / decimation_factor;
s_proc_ref = s_if_ref(1:decimation_factor:end);        % 参考通道
s_proc_filter = s_if_filter(1:decimation_factor:end);  % 滤波器通道
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc_filter);

fprintf('降采样后点数: %d\n', N_proc);

% 算法参数
win_time = 10e-6;
win_len = round(win_time * f_s_proc);
step_len = round(win_len / 8);
L_sub = round(win_len / 2);

% 存储特征
feature_f_probe = [];
feature_tau_ref = [];       % 参考通道时延
feature_tau_filter = [];    % 滤波器通道时延
feature_amp = [];

% 处理循环
num_windows = floor((N_proc - win_len) / step_len) + 1;
fprintf('处理窗口数: %d\n', num_windows);

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end
    
    % === 处理参考通道 ===
    x_window_ref = s_proc_ref(idx_start:idx_end);
    
    % === 处理滤波器通道 ===
    x_window_filter = s_proc_filter(idx_start:idx_end);
    
    % 时间-频率映射
    t_center = t_proc(idx_start + round(win_len/2));
    f_current = f_start + K * t_center;
    
    % 边缘保护
    if t_center > 0.92*T_m || t_center < 0.08*T_m, continue; end
    
    % ===== 处理参考通道 ESPRIT =====
    tau_ref_est = esprit_extract(x_window_ref, win_len, L_sub, f_s_proc, K);
    if isnan(tau_ref_est), continue; end
    
    % ===== 处理滤波器通道 ESPRIT =====
    tau_filter_est = esprit_extract(x_window_filter, win_len, L_sub, f_s_proc, K);
    if isnan(tau_filter_est), continue; end
    
    amp_est = rms(x_window_filter);
    
    feature_f_probe = [feature_f_probe, f_current];
    feature_tau_ref = [feature_tau_ref, tau_ref_est];
    feature_tau_filter = [feature_tau_filter, tau_filter_est];
    feature_amp = [feature_amp, amp_est];
end

fprintf('ESPRIT完成，有效数据点: %d\n', length(feature_f_probe));

%% 7. 结果可视化

% 计算相对时延（类似LM.m中的 tau_relative_meas = feature_tau_absolute - tau_air）
tau_relative = feature_tau_filter - feature_tau_ref;

% 理论曲线
f_theory = linspace(f_start, f_end, 200);
tau_theory = calculate_filter_group_delay(f_theory, F0_true, BW_true, N_true);

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 500]);

valid_idx = tau_relative > 0 & tau_relative < 2e-9;
scatter(feature_f_probe(valid_idx)/1e9, tau_relative(valid_idx)*1e9, 25, 'b', 'filled', ...
    'DisplayName', 'ESPRIT测量点');
hold on;
plot(f_theory/1e9, tau_theory*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '理论曲线');

xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title_str = sprintf('LFMCW滤波器群时延测量 | F0=%.1fGHz, BW=%.1fGHz, N=%d', F0_true/1e9, BW_true/1e9, N_true);
title(title_str, 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast');
xlim([f_start/1e9, f_end/1e9]);
grid on;

%% 8. 三参数LM反演
fprintf('\n开始三参数LM反演...\n');

% 数据筛选
fit_mask = (feature_f_probe >= f_start + 0.1*B) & ...
           (feature_f_probe <= f_end - 0.1*B) & ...
           (tau_relative > 0) & ...
           (tau_relative < 2e-9);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative(fit_mask);
W_raw = feature_amp(fit_mask);

if isempty(X_fit)
    warning('有效数据点为空，跳过反演');
else
    % 权重（边缘增强）
    tau_norm = Y_fit / max(Y_fit);
    edge_enhance = 1 + 0.3 * (1 - tau_norm).^2;
    Weights = ((W_raw / max(W_raw)) .* edge_enhance).^2;
    
    % 初始猜测
    F0_guess = 13.5e9;
    BW_guess = 9e9;
    N_guess = 4;
    
    fprintf('初始猜测: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', F0_guess/1e9, BW_guess/1e9, N_guess);
    
    % 归一化
    scale_F0 = 1e10;
    scale_BW = 1e10;
    scale_N = 1;
    param_init = [F0_guess/scale_F0, BW_guess/scale_BW, N_guess];
    
    % 残差函数
    ResidualFunc = @(p) sqrt(Weights) .* ...
        (calculate_filter_group_delay(X_fit, p(1)*scale_F0, p(2)*scale_BW, p(3)) - Y_fit) * 1e9;
    
    % 约束边界
    lb = [10e9/scale_F0, 4e9/scale_BW, 2];
    ub = [18e9/scale_F0, 12e9/scale_BW, 10];
    
    % 优化
    options = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', ...
        'Display', 'iter', 'MaxIterations', 100);
    
    [param_opt, ~] = lsqnonlin(ResidualFunc, param_init, lb, ub, options);
    
    F0_opt = param_opt(1) * scale_F0;
    BW_opt = param_opt(2) * scale_BW;
    N_opt = param_opt(3);
    
    % 结果输出
    fprintf('\n===== 反演结果 =====\n');
    fprintf('真实值: F0=%.4fGHz, BW=%.4fGHz, N=%.1f\n', F0_true/1e9, BW_true/1e9, N_true);
    fprintf('反演值: F0=%.4fGHz, BW=%.4fGHz, N=%.2f\n', F0_opt/1e9, BW_opt/1e9, N_opt);
    
    err_F0 = (F0_opt - F0_true)/F0_true*100;
    err_BW = (BW_opt - BW_true)/BW_true*100;
    err_N = (N_opt - N_true)/N_true*100;
    fprintf('误差: F0=%.2f%%, BW=%.2f%%, N=%.2f%%\n', err_F0, err_BW, err_N);
    fprintf('整数阶数: N=%d\n', round(N_opt));
    fprintf('====================\n');
    
    % 拟合曲线可视化
    figure(3); clf;
    set(gcf, 'Color', 'w', 'Position', [100 100 900 500]);
    
    scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
    colorbar; ylabel(colorbar, '权重');
    hold on;
    plot(f_theory/1e9, tau_theory*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实');
    tau_fit = calculate_filter_group_delay(f_theory, F0_opt, BW_opt, N_opt);
    plot(f_theory/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演');
    
    xlabel('频率 (GHz)', 'FontName', 'SimHei');
    ylabel('群时延 (ns)', 'FontName', 'SimHei');
    title_str = sprintf('反演结果 | F0误差:%.2f%%, BW误差:%.2f%%, N误差:%.2f%%', err_F0, err_BW, err_N);
    title(title_str, 'FontName', 'SimHei');
    legend('测量数据', '真实曲线', '反演曲线', 'Location', 'northeast');
    grid on;
end

fprintf('\n仿真完成！\n');

%% 局部函数

function tau_g = calculate_filter_group_delay(f_vec, F0, BW, N)
    x = (f_vec - F0) / (BW/2);
    tau_g = (2*N) / (pi*BW) .* (1 + x.^2).^(-(N+1)/2);
end

function tau_est = esprit_extract(x_window, win_len, L_sub, f_s_proc, K)
    % ESPRIT差频提取辅助函数
    % 返回: tau_est - 估计的时延 (如果失败返回NaN)
    
    M_sub = win_len - L_sub + 1;
    
    % Hankel矩阵
    X_hankel = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end
    
    % 协方差矩阵（前后向平均）
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;
    
    % 特征分解
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
        if g_mean > 0 && a_mean > 0
            mdl_cost(k_mdl+1) = -(p-k_mdl)*N_snaps*log(g_mean/a_mean) + 0.5*k_mdl*(2*p-k_mdl)*log(N_snaps);
        else
            mdl_cost(k_mdl+1) = inf;
        end
    end
    [~, min_idx] = min(mdl_cost);
    num_sources = max(1, min(min_idx-1, 3));
    
    % TLS-ESPRIT
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    % 频率筛选
    valid_mask = (est_freqs > 20e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs)
        tau_est = NaN;
        return;
    end
    
    [f_beat_est, ~] = min(valid_freqs);
    tau_est = f_beat_est / K;
end
