%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW滤波器诊断系统 
% 关键修正：使用理论公式直接在频域生成滤波器响应
% 解决问题：Butterworth滤波器与Lorentzian公式的模型失配
% 创建日期：2026-01-15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. 仿真参数设置

fprintf('===== LFMCW滤波器诊断系统）=====\n'); 

% =========================================================================
% LFMCW雷达参数（增加扫频时间以提高分辨率）
% =========================================================================
f_start = 10e9;              % 扫频起始频率 (Hz)
f_end = 18e9;                % 扫频终止频率 (Hz)
T_m = 100e-6;                % 调制周期 (s) - 增加以降低K
B_sweep = f_end - f_start;   % 扫频带宽 (Hz)
K = B_sweep/T_m;             % 调频斜率 (Hz/s)
f_s = 40e9;                  % 采样频率 (Hz)

fprintf('调频斜率 K = %.2f MHz/μs\n', K/1e12);

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

fprintf('采样点数: %d (%.1f M点)\n', N_samples, N_samples/1e6);
fprintf('LFMCW参数: %.1f-%.1f GHz, T_m=%.0f μs\n', f_start/1e9, f_end/1e9, T_m*1e6);
fprintf('滤波器参数: F0=%.2f GHz, BW=%.2f GHz, N=%d\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('==========================================\n\n');

%% 2. LFMCW信号生成

f_t = f_start + K*mod(t, T_m);   % 瞬时频率
phi_t = 2*pi*cumsum(f_t)*t_s;    % 瞬时相位
s_tx = cos(phi_t);               % 发射信号

fprintf('LFMCW信号生成完成\n');

%% 3. 信号传播模拟：使用理论公式直接频域建模（自洽方法）

fprintf('开始滤波器传播模拟（频域直接建模）...\n');

% 3.1 参考通道（仅延迟）
delay_samples_ref = round(tau_ref/t_s);
s_rx_ref = [zeros(1, delay_samples_ref) s_tx(1:end-delay_samples_ref)];

% 3.2 滤波器通道 - 频域建模（修正版：相位=群时延的积分）

% 构建频率轴
f_fft = (0:N_samples-1)*(f_s/N_samples);
idx_neg = f_fft >= f_s/2;
f_fft(idx_neg) = f_fft(idx_neg) - f_s;

% FFT变换
S_tx = fft(s_tx);

% ===== 关键修正：正确计算相位响应 =====
% 群时延定义：tau_g = -d(phi)/d(omega)
% 所以相位：phi(omega) = -integral(tau_g * d(omega))

% 只处理正频率部分，负频率利用对称性
f_pos = f_fft(1:floor(N_samples/2)+1);  % 正频率轴（包括DC和Nyquist）
omega_pos = 2*pi*f_pos;
d_omega = 2*pi * (f_s/N_samples);       % 频率步进

% 计算正频率的群时延
tau_pos = calculate_filter_group_delay(f_pos, F0_true, BW_true, N_true);

% 积分得到相位：phi[k] = -sum(tau[0:k] * d_omega)
% 使用cumsum实现累积积分
phi_pos = -cumsum(tau_pos) * d_omega;
phi_pos(1) = 0;  % DC相位为0

% 构造全频谱相位（负频率相位 = -正频率相位，保证实信号）
if mod(N_samples, 2) == 0
    % 偶数采样点
    phi_full = [phi_pos, -fliplr(phi_pos(2:end-1))];
else
    % 奇数采样点
    phi_full = [phi_pos, -fliplr(phi_pos(2:end))];
end

% =========================================================================
% 幅度响应建模（工程级 - 引入非理想特性）
% 参考Datasheet：纹波≤1.2dB, 阻带抑制≥80dB
% =========================================================================

% 1. 基础 Butterworth 幅度（理想模型）
x_norm = (abs(f_fft) - F0_true) / (BW_true/2);
H_mag_ideal = (1 + x_norm.^2).^(-N_true/2);

% 2. 通带纹波模拟 (Ripple ≤ 1.2dB)
% 在通带内叠加随机波动，模拟非理想谐振腔耦合
ripple_db = 1.2;
ripple_linear = 10^(ripple_db/20) - 1;  % dB转线性
passband_mask = abs(x_norm) <= 1;  % 通带定义
% 使用固定种子确保可重复
rng(123);
ripple_factor = 1 + (rand(size(f_fft)) - 0.5) * 2 * ripple_linear .* passband_mask;

% 3. 阻带抑制与系统底噪 (Rejection ≥ 80dB)
% 设定系统动态范围（考虑雷达接收机底噪）
dynamic_range_db = 90;  % 典型雷达系统动态范围
noise_floor_linear = 10^(-dynamic_range_db/20);

% 4. 插入损耗 (Insertion Loss ≤ 2.0dB)
IL_db = 2.0;
IL_linear = 10^(-IL_db/20);  % 整体衰减因子

% 合成真实幅度响应
H_mag_real = IL_linear * H_mag_ideal .* ripple_factor;

% 限制最小幅度（模拟阻带抑制）
H_mag_real = max(H_mag_real, noise_floor_linear);

fprintf('  非理想特性已加入: 纹波=%.1fdB, 阻带抑制=%.0fdB, 插损=%.1fdB\n', ...
    ripple_db, dynamic_range_db, IL_db);

% 参考时延的相位
omega_full = 2*pi*f_fft;
H_ref = exp(-1i * omega_full * tau_ref);

% 组合传递函数（幅度×相位×参考延迟）
H_filter = H_mag_real .* exp(1i * phi_full) .* H_ref;

% 5. 加入加性高斯白噪声 (AWGN) - 模拟系统热噪声
% 这决定了低幅度区域的信噪比！
awgn_level = noise_floor_linear * 0.1;  % 底噪的10%
awgn = awgn_level * (randn(size(f_fft)) + 1i*randn(size(f_fft))) / sqrt(2);
H_filter = H_filter + awgn;

H_filter(1) = 0;  % 直流分量置零

% 应用传递函数
S_rx_filter = S_tx .* H_filter;
s_rx_filter = real(ifft(S_rx_filter));

fprintf('滤波器传播完成（修正：相位=群时延积分）\n');

%% 4. 混频处理与差频信号提取

% 4.1 参考通道混频
s_mix_ref = s_tx .* s_rx_ref;

% 低通滤波器
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if_ref = filtfilt(b_lp, a_lp, s_mix_ref);

% 4.2 滤波器通道混频
s_mix_filter = s_tx .* s_rx_filter;
s_if_filter = filtfilt(b_lp, a_lp, s_mix_filter);

fprintf('混频处理完成\n');

%% 5. LFMCW信号处理流程可视化（参考LM.m的Figure 1-8）

fprintf('\n生成信号处理流程图像...\n');

% 辅助变量
t_display = min(5e-6, T_m);
idx_display = round(t_display/t_s);
f_range = [f_start-0.5e9, f_end+0.5e9];
f_indices = find(f_fft >= f_range(1) & f_fft <= f_range(2));

% --- Figure 1: 时域对比（发射 vs 接收）---
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);
subplot(1,2,1);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', 'LineWidth', 1);
hold on;
plot(t(1:idx_display)*1e6, s_rx_ref(1:idx_display), 'r--', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 1a: 发射信号 vs 参考接收', 'FontName', 'SimHei');
legend('发射', '参考接收');
grid on;

subplot(1,2,2);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', 'LineWidth', 1);
hold on;
plot(t(1:idx_display)*1e6, s_rx_filter(1:idx_display), 'r--', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 1b: 发射信号 vs 滤波器接收', 'FontName', 'SimHei');
legend('发射', '滤波器接收');
grid on;

% --- Figure 2: 频域对比 ---
figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);
S_TX_mag = abs(S_tx);
S_RX_ref_mag = abs(fft(s_rx_ref));
S_RX_filter_mag = abs(fft(s_rx_filter));

subplot(1,2,1);
plot(f_fft(f_indices)/1e9, S_TX_mag(f_indices), 'b', 'LineWidth', 1);
hold on;
plot(f_fft(f_indices)/1e9, S_RX_ref_mag(f_indices), 'r--', 'LineWidth', 1);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 2a: 发射 vs 参考接收（频谱）', 'FontName', 'SimHei');
legend('发射', '参考接收');
grid on;

subplot(1,2,2);
plot(f_fft(f_indices)/1e9, S_TX_mag(f_indices), 'b', 'LineWidth', 1);
hold on;
plot(f_fft(f_indices)/1e9, S_RX_filter_mag(f_indices), 'r--', 'LineWidth', 1);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 2b: 发射 vs 滤波器接收（频谱）', 'FontName', 'SimHei');
legend('发射', '滤波器接收');
grid on;

% --- Figure 3: 差频信号时域 ---
figure(3); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);
t_if_disp = min(20e-6, T_m);
idx_if = round(t_if_disp/t_s);

subplot(1,2,1);
plot(t(1:idx_if)*1e6, s_if_ref(1:idx_if), 'b', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 3a: 参考通道差频（时域）', 'FontName', 'SimHei');
grid on;

subplot(1,2,2);
plot(t(1:idx_if)*1e6, s_if_filter(1:idx_if), 'b', 'LineWidth', 1);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 3b: 滤波器通道差频（时域）', 'FontName', 'SimHei');
grid on;

% --- Figure 4: 差频信号频域 ---
figure(4); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);
S_IF_ref = abs(fft(s_if_ref));
S_IF_filter = abs(fft(s_if_filter));
f_if_lim = 400e3;  % 缩小到400 kHz，让信号看起来更饱满
idx_if_f = round(f_if_lim/(f_s/N_samples));

f_if_axis = (0:N_samples-1)*(f_s/N_samples);

subplot(1,2,1);
stem(f_if_axis(1:idx_if_f)/1e3, S_IF_ref(1:idx_if_f), 'b', 'MarkerSize', 2);
xlabel('频率 (kHz)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 4a: 参考通道差频（频谱）', 'FontName', 'SimHei');
xlim([0 400]);  % 限制到0-400 kHz
grid on;

subplot(1,2,2);
stem(f_if_axis(1:idx_if_f)/1e3, S_IF_filter(1:idx_if_f), 'b', 'MarkerSize', 2);
xlabel('频率 (kHz)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 4b: 滤波器通道差频（频谱）', 'FontName', 'SimHei');
xlim([0 400]);  % 限制到0-400 kHz
grid on;

fprintf('信号处理流程图像生成完成（Figure 1-4）\n');

%% 6. 滑动窗口 + ESPRIT 特征提取

fprintf('\n开始ESPRIT特征提取...\n');

% 降采样
decimation_factor = 150;
f_s_proc = f_s / decimation_factor;
s_proc_ref = s_if_ref(1:decimation_factor:end);
s_proc_filter = s_if_filter(1:decimation_factor:end);
t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc_filter);

fprintf('降采样后点数: %d\n', N_proc);

% 算法参数
win_time = 8e-6;  % 减小窗口时间以提高频率分辨率
win_len = round(win_time * f_s_proc);
step_len = round(win_len / 8);
L_sub = round(win_len / 2);

% 窗口内频率跨度检查
delta_f_win = K * win_time;
fprintf('窗口内频率跨度: %.2f GHz (应 << BW=%.1f GHz)\n', delta_f_win/1e9, BW_true/1e9);

% 存储特征
feature_f_probe = [];
feature_tau_ref = [];
feature_tau_filter = [];
feature_amplitude = [];

% 处理循环
num_windows = floor((N_proc - win_len) / step_len) + 1;
fprintf('处理窗口数: %d\n', num_windows);

hWait = waitbar(0, 'ESPRIT特征提取中...');

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end
    
    x_window_ref = s_proc_ref(idx_start:idx_end);
    x_window_filter = s_proc_filter(idx_start:idx_end);
    
    % 时间-频率映射
    t_center = t_proc(idx_start + round(win_len/2));
    f_current = f_start + K * t_center;
    
    % 边缘保护
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % ESPRIT处理
    tau_ref_est = esprit_extract(x_window_ref, win_len, L_sub, f_s_proc, K);
    if isnan(tau_ref_est), continue; end
    
    tau_filter_est = esprit_extract(x_window_filter, win_len, L_sub, f_s_proc, K);
    if isnan(tau_filter_est), continue; end
    
    amp_est = rms(x_window_filter);
    
    feature_f_probe = [feature_f_probe, f_current];
    feature_tau_ref = [feature_tau_ref, tau_ref_est];
    feature_tau_filter = [feature_tau_filter, tau_filter_est];
    feature_amplitude = [feature_amplitude, amp_est];
    
    if mod(i, 20) == 0, waitbar(i/num_windows, hWait); end
end
close(hWait);

fprintf('ESPRIT完成，有效数据点: %d\n', length(feature_f_probe));

%% 7. 计算相对时延

tau_relative_meas = feature_tau_filter - feature_tau_ref;

%% 8. 可视化测量结果 vs 理论曲线（Figure 5）

f_theory = linspace(f_start, f_end, 200);
tau_theory = calculate_filter_group_delay(f_theory, F0_true, BW_true, N_true);

figure(5); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 600]);

valid_idx = tau_relative_meas > 0 & tau_relative_meas < 3e-9;

scatter(feature_f_probe(valid_idx)/1e9, tau_relative_meas(valid_idx)*1e9, 25, 'b', 'filled', ...
    'DisplayName', 'ESPRIT测量点');
hold on;
plot(f_theory/1e9, tau_theory*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '理论曲线');

xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title_str = sprintf('Figure 5: LFMCW群时延测量验证 | F0=%.1fGHz, BW=%.1fGHz, N=%d', ...
    F0_true/1e9, BW_true/1e9, N_true);
title(title_str, 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast');
xlim([f_start/1e9, f_end/1e9]);
grid on;

%% 9. 三参数LM反演

fprintf('\n=============================================\n');
fprintf('开始三参数反演 (Weighted LM Algorithm)...\n');

% 数据筛选
fit_mask = (feature_f_probe >= f_start + 0.05*B_sweep) & ...
           (feature_f_probe <= f_end - 0.05*B_sweep) & ...
           (tau_relative_meas > 1e-11);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);

if isempty(X_fit)
    error('有效拟合数据点为空！');
end

% 权重
Weights_base = (W_raw / max(W_raw)).^2;
tau_norm = Y_fit / max(Y_fit);
edge_factor = 1 + 0.5 * (1 - tau_norm).^2;
Weights = Weights_base .* edge_factor;
Weights = Weights / max(Weights);

fprintf('有效数据点: %d\n', length(X_fit));

% =========================================================================
% 初始值策略（智能版 - 基于数据特征提取）
% 参考LM.m的哲学：从物理特征中提取先验信息
% =========================================================================

% 1. F0猜测：使用测量数据的"峰值位置"
[~, peak_idx] = max(Y_fit);  % 找到群时延最大的点
F0_guess = X_fit(peak_idx);  % 该点即为近似中心频率

% 2. BW猜测：基于半高宽（FWHM）估计
half_max = max(Y_fit) / 2;
idx_above_half = find(Y_fit >= half_max);
if length(idx_above_half) >= 2
    BW_guess = 2 * (X_fit(idx_above_half(end)) - X_fit(idx_above_half(1)));
else
    BW_guess = 1.2 * (max(X_fit) - min(X_fit));  % 回退策略
end

% 3. N猜测：基于边缘陡峭度估计
% 滤波器越高阶，边缘越陡
edge_points = Y_fit(Y_fit < 0.3*max(Y_fit));
if ~isempty(edge_points)
    edge_slope = std(edge_points);  % 边缘波动越小，阶数越高
    N_guess = round(3 + 2/edge_slope);  % 简化启发式
    N_guess = max(2, min(N_guess, 8));  % 限制范围[2,8]
else
    N_guess = 4;  % 回退策略
end

fprintf('智能初始猜测: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', ...
    F0_guess/1e9, BW_guess/1e9, N_guess);

% 归一化
scale_F0 = 1e10;
scale_BW = 1e10;
scale_N = 1;
param_init = [F0_guess/scale_F0, BW_guess/scale_BW, N_guess/scale_N];

% 残差函数
ResidualFunc = @(p) WeightedResiduals_Filter3P(...
    p, scale_F0, scale_BW, scale_N, X_fit, Y_fit, Weights);

% 优化
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'iter', ...
    'StepTolerance', 1e-10, ...
    'FunctionTolerance', 1e-10, ...
    'DiffMinChange', 0.001, ...
    'MaxIterations', 200);

[param_opt, ~, ~, exitflag] = lsqnonlin(ResidualFunc, param_init, [], [], options);

F0_opt = param_opt(1) * scale_F0;
BW_opt = param_opt(2) * scale_BW;
N_opt = param_opt(3) * scale_N;

%% 10. 结果输出

fprintf('\n===== 三参数反演结果 =====\n');
fprintf('真实值: F0=%.4fGHz, BW=%.4fGHz, N=%.1f\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('反演值: F0=%.4fGHz, BW=%.4fGHz, N=%.2f\n', F0_opt/1e9, BW_opt/1e9, N_opt);

err_F0 = (F0_opt - F0_true)/F0_true*100;
err_BW = (BW_opt - BW_true)/BW_true*100;
err_N = (N_opt - N_true)/N_true*100;

fprintf('相对误差: F0=%.2f%%, BW=%.2f%%, N=%.2f%%\n', err_F0, err_BW, err_N);
fprintf('整数阶数: N=%d\n', round(N_opt));
fprintf('===========================\n');

%% 11. 可视化拟合结果（Figure 6）

figure(6); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 700]);

subplot(3,1,1);
scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
colormap(jet); colorbar; ylabel(colorbar, '权重', 'FontName', 'SimHei');
hold on;
plot(f_theory/1e9, tau_theory*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实曲线');
tau_fit = calculate_filter_group_delay(f_theory, F0_opt, BW_opt, N_opt);
plot(f_theory/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演曲线');
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title_str = sprintf('Figure 6: 反演结果 | F0误差:%.2f%%, BW误差:%.2f%%, N误差:%.2f%%', err_F0, err_BW, err_N);
title(title_str, 'FontName', 'SimHei');
legend('测量数据', '真实曲线', '反演曲线', 'Location', 'northeast');
grid on;

subplot(3,1,2);
residuals = Y_fit - calculate_filter_group_delay(X_fit, F0_opt, BW_opt, N_opt);
stem(X_fit/1e9, residuals*1e9, 'b', 'MarkerSize', 2);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontName', 'SimHei');
title('残差分布', 'FontName', 'SimHei');
grid on; yline(0, 'r--');

subplot(3,1,3);
plot(X_fit/1e9, Weights, 'k-', 'LineWidth', 1.5);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('归一化权重', 'FontName', 'SimHei');
title('权重分布', 'FontName', 'SimHei');
grid on; ylim([0 1.1]);

fprintf('\n仿真完成！\n');

%% 局部函数

function tau_g = calculate_filter_group_delay(f_vec, F0, BW, N)
    x = (f_vec - F0) / (BW/2);
    tau_g = (2*N) / (pi*BW) .* (1 + x.^2).^(-(N+1)/2);
end

function tau_est = esprit_extract(x_window, win_len, L_sub, f_s_proc, K)
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
    
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));
    
    valid_mask = (est_freqs > 50e3) & (est_freqs < 10e6);
    valid_freqs = est_freqs(valid_mask);
    
    if isempty(valid_freqs)
        tau_est = NaN;
        return;
    end
    
    [f_beat_est, ~] = min(valid_freqs);
    tau_est = f_beat_est / K;
end

function F_vec = WeightedResiduals_Filter3P(p_scaled, scale_F0, scale_BW, scale_N, f_data, tau_data, weights)
    F0_val = p_scaled(1) * scale_F0;
    BW_val = p_scaled(2) * scale_BW;
    N_val = p_scaled(3) * scale_N;
    
    if F0_val <= 0 || BW_val <= 0 || N_val <= 0.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    if F0_val < min(f_data)*0.5 || F0_val > max(f_data)*1.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    if BW_val < 1e9 || BW_val > 20e9 || N_val < 1 || N_val > 15
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    try
        tau_theory = calculate_filter_group_delay(f_data, F0_val, BW_val, N_val);
        F_vec = sqrt(weights) .* (tau_theory - tau_data) * 1e9;
        
        if any(isnan(F_vec)) || any(isinf(F_vec))
            F_vec = ones(size(f_data)) * 1e5;
        end
    catch
        F_vec = ones(size(f_data)) * 1e5;
    end
end
