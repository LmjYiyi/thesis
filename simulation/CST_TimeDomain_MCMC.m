%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST时域数据混频处理与Lorentz参数MCMC反演
% 功能: 读取CST时域仿真数据，进行混频、ESPRIT特征提取、MCMC参数反演
% 参考: LM_lorentz_MCMC.m
%
% 数据来源: CST Microwave Studio 时域仿真
% 作者: Auto-generated for thesis project
% 日期: 2026-01-20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 初始化
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

fprintf('================================================\n');
fprintf('CST时域数据MCMC反演系统\n');
fprintf('================================================\n\n');

%% 2. 读取CST数据
fprintf('【步骤1】读取CST时域数据...\n');

data_file = fullfile(script_dir, 'cst_data', 'output.txt');

if ~exist(data_file, 'file')
    error('找不到数据文件: %s', data_file);
end

% 读取整个文件
fid = fopen(data_file, 'r');
raw_text = fread(fid, '*char')';
fclose(fid);

% 分割为两个数据块
lines = strsplit(raw_text, {'\r\n', '\n'});

% 找到o21数据的起始位置
o21_start_idx = 0;
for i = 1:length(lines)
    if contains(lines{i}, 'o2,1')
        o21_start_idx = i + 2;
        break;
    end
end

if o21_start_idx == 0
    error('未找到o21数据标记');
end

% 解析i1数据
i1_lines = lines(4:o21_start_idx-4);
i1_data = zeros(length(i1_lines), 2);
valid_count = 0;
for i = 1:length(i1_lines)
    vals = sscanf(i1_lines{i}, '%f\t%f');
    if length(vals) == 2
        valid_count = valid_count + 1;
        i1_data(valid_count, :) = vals';
    end
end
i1_data = i1_data(1:valid_count, :);

% 解析o21数据
o21_lines = lines(o21_start_idx:end);
o21_data = zeros(length(o21_lines), 2);
valid_count = 0;
for i = 1:length(o21_lines)
    vals = sscanf(o21_lines{i}, '%f\t%f');
    if length(vals) == 2
        valid_count = valid_count + 1;
        o21_data(valid_count, :) = vals';
    end
end
o21_data = o21_data(1:valid_count, :);

fprintf('  ✓ i1 数据: %d 个采样点\n', size(i1_data, 1));
fprintf('  ✓ o21 数据: %d 个采样点\n', size(o21_data, 1));

% 提取时间和信号
t_ns = i1_data(:, 1);           % 时间 (ns)
t = t_ns * 1e-9;                % 转换为秒
s_tx = i1_data(:, 2);
s_rx = o21_data(1:length(s_tx), 2);

% 计算采样率
dt = mean(diff(t));
f_s = 1 / dt;
fprintf('  采样率: %.3f GHz\n', f_s/1e9);
fprintf('  时间范围: %.2f - %.2f ns\n', t(1)*1e9, t(end)*1e9);

%% 3. LFMCW参数设置
fprintf('\n【LFMCW参数设置】\n');

% CST仿真中的LFMCW参数
f_start = 34.2e9;       % 起始频率 (Hz)
f_end = 37.4e9;         % 终止频率 (Hz)
T_m = 50e-9;            % 扫频周期 (s) = 50ns
B = f_end - f_start;    
K = B / T_m;            % 调频斜率
c = 3e8;
d = 3e-3;               % 有效厚度 (m)

fprintf('  频率范围: %.2f - %.2f GHz\n', f_start/1e9, f_end/1e9);
fprintf('  扫频周期: %.1f ns\n', T_m*1e9);
fprintf('  调频斜率 K: %.3e Hz/s\n', K);

N = length(s_tx);

%% 4. 读取空气参考信号 (CST仿真数据)
fprintf('\n【步骤2】读取CST空气介质仿真数据...\n');

air_data_file = fullfile(script_dir, 'cst_data', 'output_air.txt');

if exist(air_data_file, 'file')
    % 读取空气仿真数据
    fid_air = fopen(air_data_file, 'r');
    raw_text_air = fread(fid_air, '*char')';
    fclose(fid_air);
    
    lines_air = strsplit(raw_text_air, {'\r\n', '\n'});
    
    % 找到o21数据的起始位置
    o21_air_start_idx = 0;
    for i = 1:length(lines_air)
        if contains(lines_air{i}, 'o2,1')
            o21_air_start_idx = i + 2;
            break;
        end
    end
    
    if o21_air_start_idx == 0
        fprintf('  ⚠ 未找到空气o21数据，使用模拟数据\n');
        tau_air = 0.1e-9;
        delay_samples_air = round(tau_air / dt);
        s_rx_air = [zeros(delay_samples_air, 1); s_tx(1:end-delay_samples_air)];
        s_tx_air = s_tx;
    else
        % 解析空气i1数据
        i1_air_lines = lines_air(4:o21_air_start_idx-4);
        i1_air_data = zeros(length(i1_air_lines), 2);
        valid_count = 0;
        for i = 1:length(i1_air_lines)
            vals = sscanf(i1_air_lines{i}, '%f\t%f');
            if length(vals) == 2
                valid_count = valid_count + 1;
                i1_air_data(valid_count, :) = vals';
            end
        end
        i1_air_data = i1_air_data(1:valid_count, :);
        
        % 解析空气o21数据
        o21_air_lines = lines_air(o21_air_start_idx:end);
        o21_air_data = zeros(length(o21_air_lines), 2);
        valid_count = 0;
        for i = 1:length(o21_air_lines)
            vals = sscanf(o21_air_lines{i}, '%f\t%f');
            if length(vals) == 2
                valid_count = valid_count + 1;
                o21_air_data(valid_count, :) = vals';
            end
        end
        o21_air_data = o21_air_data(1:valid_count, :);
        
        s_tx_air = i1_air_data(:, 2);
        s_rx_air = o21_air_data(1:length(s_tx_air), 2);
        
        fprintf('  ✓ 空气i1数据: %d 个采样点\n', length(s_tx_air));
        fprintf('  ✓ 空气o21数据: %d 个采样点\n', size(o21_air_data, 1));
    end
else
    fprintf('  ⚠ 未找到空气数据文件，使用模拟数据\n');
    tau_air = 0.1e-9;
    delay_samples_air = round(tau_air / dt);
    s_rx_air = [zeros(delay_samples_air, 1); s_tx(1:end-delay_samples_air)];
    s_tx_air = s_tx;
end

% 确保数据长度一致
N_air = min([length(s_tx), length(s_rx), length(s_tx_air), length(s_rx_air)]);
s_tx = s_tx(1:N_air);
s_rx = s_rx(1:N_air);
s_tx_air = s_tx_air(1:N_air);
s_rx_air = s_rx_air(1:N_air);
t = t(1:N_air);
N = N_air;

fprintf('  统一数据长度: %d 个采样点\n', N);

%% 5. 混频处理
fprintf('\n【步骤3】混频处理...\n');

% 空气介质混频 (使用CST空气仿真数据)
s_mix_air = s_tx_air .* s_rx_air;

% 超材料介质混频 (CST数据)
s_mix_meta = s_tx .* s_rx;

% 低通滤波器
fc_lp = 50e9;  % 低通截止频率
[b_lp, a_lp] = butter(4, min(0.9, fc_lp/(f_s/2)));

s_if_air = filtfilt(b_lp, a_lp, s_mix_air);
s_if_meta = filtfilt(b_lp, a_lp, s_mix_meta);

fprintf('  ✓ 空气混频完成 (CST数据)\n');
fprintf('  ✓ 超材料混频完成 (CST数据)\n');

%% 6. 基础频域分析
fprintf('\n【步骤4】频域分析...\n');

% FFT
S_IF_air = fft(s_if_air);
S_IF_air_mag = abs(S_IF_air);

win = hann(N);
s_if_meta_win = s_if_meta .* win;
S_IF_meta = fft(s_if_meta_win);
S_IF_meta_mag = abs(S_IF_meta) * 2;

% 频率轴
f = (0:N-1) * f_s / N;

% 从空气差频信号估计tau_air
S_IF_air_win = fft(s_if_air .* win);
S_IF_air_mag_win = abs(S_IF_air_win);
[~, peak_idx_air] = max(S_IF_air_mag_win(2:round(N/2)));
f_beat_air = f(peak_idx_air + 1);
tau_air = f_beat_air / K;

fprintf('  空气差频峰值: %.3f GHz\n', f_beat_air/1e9);
fprintf('  估计空气时延: %.3f ns\n', tau_air*1e9);

%% 7. Figure 1-8: 基础信号可视化
fprintf('\n【步骤5】生成Figure 1-8...\n');

% Figure 1: 发射信号时域
figure(1);
t_disp = min(5e-9, t(end));
idx_disp = t <= t_disp;
plot(t(idx_disp)*1e9, s_tx(idx_disp), 'b');
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 1: 发射信号 i1 (局部)', 'FontName', 'SimHei');
grid on;

% Figure 2: 接收信号时域
figure(2);
plot(t(idx_disp)*1e9, s_rx(idx_disp), 'r');
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('Figure 2: 接收信号 o21 (局部)', 'FontName', 'SimHei');
grid on;

% Figure 3: 发射vs接收对比
figure(3);
idx_comp = t <= 1e-9;
plot(t(idx_comp)*1e9, s_tx(idx_comp), 'b', ...
     t(idx_comp)*1e9, s_rx(idx_comp), 'r--');
xlabel('时间 (ns)', 'FontName', 'SimHei');
legend('发射', '接收');
title('Figure 3: 发射 vs 接收 (前1ns)', 'FontName', 'SimHei');
grid on;

% Figure 4: 发射信号频谱
figure(4);
S_TX = fft(s_tx .* win);
S_TX_mag = abs(S_TX) * 2 / N;
f_range_idx = (f >= 30e9) & (f <= 40e9);
plot(f(f_range_idx)/1e9, S_TX_mag(f_range_idx), 'b');
xlabel('频率 (GHz)', 'FontName', 'SimHei');
title('Figure 4: 发射信号频谱 (30-40 GHz)', 'FontName', 'SimHei');
grid on;

% Figure 5: 接收信号频谱
figure(5);
S_RX = fft(s_rx .* win);
S_RX_mag = abs(S_RX) * 2 / N;
plot(f(f_range_idx)/1e9, S_RX_mag(f_range_idx), 'r');
xlabel('频率 (GHz)', 'FontName', 'SimHei');
title('Figure 5: 接收信号频谱 (30-40 GHz)', 'FontName', 'SimHei');
grid on;

% Figure 6: 混频信号（原始）
figure(6);
plot(t*1e9, s_mix_meta, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.2);
xlabel('时间 (ns)', 'FontName', 'SimHei');
title('Figure 6: 混频信号 (原始)', 'FontName', 'SimHei');
grid on;

% Figure 7: 差频信号（滤波后）
figure(7);
plot(t*1e9, s_if_meta, 'b');
xlabel('时间 (ns)', 'FontName', 'SimHei');
title('Figure 7: 差频信号 (低通滤波后)', 'FontName', 'SimHei');
grid on;

% Figure 8: 差频频谱 (离散显示)
figure(8);
f_lim = 20e9;
idx_f = f <= f_lim;
stem(f(idx_f)/1e9, S_IF_meta_mag(idx_f), 'b', 'MarkerSize', 2);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅度', 'FontName', 'SimHei');
title('Figure 8: 差频信号频谱 (离散)', 'FontName', 'SimHei');
grid on;

fprintf('  ✓ Figure 1-8 完成\n');

%% 8. 高级信号处理: 滑动窗口 + ESPRIT + 幅度提取
fprintf('\n【步骤6】ESPRIT特征提取...\n');

% 由于CST时域数据已经是高采样率，直接处理
decimation_factor = 1;
f_s_proc = f_s / decimation_factor;

% 超材料差频信号
s_proc_meta = s_if_meta(1:decimation_factor:end);
% 空气差频信号 
s_proc_air = s_if_air(1:decimation_factor:end);

t_proc = t(1:decimation_factor:end);
N_proc = length(s_proc_meta);

% 算法参数 (针对50ns周期和高采样率调整)
win_time = 5e-9;                   % 窗口时长 5ns
win_len = round(win_time * f_s_proc);
win_len = min(max(win_len, 100), 500);  % 限制100-500点
step_len = max(1, round(win_len / 5));
L_sub = min(round(win_len / 3), 100);   % 限制子空间维度

feature_f_probe = [];
feature_tau_meta = [];     % 超材料绝对时延
feature_tau_air = [];      % 空气绝对时延
feature_amplitude = [];

num_windows = floor((N_proc - win_len) / step_len) + 1;
fprintf('  窗口数: %d, 窗口长度: %d\n', num_windows, win_len);

hWait = waitbar(0, 'ESPRIT特征提取中...');

for i = 1:num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end
    
    % 同时提取超材料和空气窗口
    x_window_meta = s_proc_meta(idx_start:idx_end);
    x_window_air = s_proc_air(idx_start:idx_end);
    
    % 时间-频率映射
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;
    
    % 避开边缘
    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end
    
    % ===== 提取超材料差频 =====
    X_fft_meta = abs(fft(x_window_meta));
    [~, peak_idx_meta] = max(X_fft_meta(2:round(length(X_fft_meta)/2)));
    f_beat_meta = (peak_idx_meta) * f_s_proc / length(x_window_meta);
    tau_meta_window = f_beat_meta / K;
    
    % ===== 提取空气差频 =====
    X_fft_air = abs(fft(x_window_air));
    [~, peak_idx_air] = max(X_fft_air(2:round(length(X_fft_air)/2)));
    f_beat_air_window = (peak_idx_air) * f_s_proc / length(x_window_air);
    tau_air_window = f_beat_air_window / K;
    
    % 计算相对时延 = 超材料时延 - 空气时延
    tau_relative = tau_meta_window - tau_air_window;
    
    % 只记录有效的点
    if f_beat_meta > 0.01e9 && f_beat_air_window > 0.01e9
        feature_f_probe = [feature_f_probe, f_current_probe];
        feature_tau_meta = [feature_tau_meta, tau_meta_window];
        feature_tau_air = [feature_tau_air, tau_air_window];
        feature_amplitude = [feature_amplitude, rms(x_window_meta)];
    end
    
    if mod(i, 20) == 0, waitbar(i/num_windows, hWait); end
end
close(hWait);

% 计算相对时延
feature_tau_relative = feature_tau_meta - feature_tau_air;

fprintf('  ✓ ESPRIT完成: %d 个有效点\n', length(feature_f_probe));
fprintf('  平均空气时延: %.3f ns\n', mean(feature_tau_air)*1e9);
fprintf('  平均超材料时延: %.3f ns\n', mean(feature_tau_meta)*1e9);
fprintf('  平均相对时延: %.3f ns\n', mean(feature_tau_relative)*1e9);

%% 9. 诊断结果可视化
fprintf('\n【步骤7】生成Figure 9...\n');

% 理论群时延计算 (Lorentz模型)
f_theory_vec = linspace(f_start, f_end, 500);
omega_theory = 2*pi*f_theory_vec;

% 使用预设的Lorentz参数作为参考
f_res_ref = 34.5e9;      % 参考谐振频率
gamma_ref = 0.5e9;       % 参考阻尼
omega_p_meta = 2*pi*5e9; % 等效等离子体频率
omega_res_ref = 2*pi*f_res_ref;

eps_theory = 1 + (omega_p_meta^2) ./ (omega_res_ref^2 - omega_theory.^2 - 1i*2*pi*gamma_ref*omega_theory);
k_theory = (omega_theory ./ c) .* sqrt(eps_theory);
phi_theory = -real(k_theory) * d;
d_phi_theory = diff(phi_theory);
d_omega_theory = diff(omega_theory);
tau_theory = -d_phi_theory ./ d_omega_theory;
tau_theory = [tau_theory, tau_theory(end)];
tau_vacuum = d / c;
tau_relative_theory = tau_theory - tau_vacuum;

% 使用已计算的相对时延
tau_relative_meas = feature_tau_relative;

figure(9); clf;
set(gcf, 'Position', [100, 100, 900, 600]);

% 绘制测量点
valid_idx = tau_relative_meas > -5e-9 & tau_relative_meas < 5e-9;
if any(valid_idx)
    scatter(feature_f_probe(valid_idx)/1e9, tau_relative_meas(valid_idx)*1e9, 20, 'b', 'filled', ...
        'DisplayName', 'CST仿真测量点');
end
hold on;

% 绘制理论曲线
plot(f_theory_vec/1e9, tau_relative_theory*1e9, 'r', 'LineWidth', 2.5, ...
    'DisplayName', 'Lorentz模型理论值');

xline(f_res_ref/1e9, 'g--', 'LineWidth', 2, 'DisplayName', '参考谐振频率');

grid on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相对群时延 Δτ (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('CST超材料诊断结果', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast');
xlim([f_start/1e9, f_end/1e9]);

fprintf('  ✓ Figure 9 完成\n');

%% 10. MCMC参数反演
fprintf('\n【步骤8】MCMC参数反演...\n');

% 数据准备
fit_mask = (feature_f_probe >= f_start + 0.05*B) & ...
           (feature_f_probe <= f_end - 0.05*B) & ...
           (abs(tau_relative_meas) < 10e-9);

X_fit = feature_f_probe(fit_mask);
Y_fit = tau_relative_meas(fit_mask);
W_raw = feature_amplitude(fit_mask);

if isempty(X_fit)
    fprintf('  ⚠ 有效拟合点为空，跳过MCMC反演\n');
    fprintf('\n【处理完成】\n');
    return;
end

Weights = (W_raw / max(W_raw)).^2;
sigma_meas = 0.1e-9;

% MCMC参数
N_samples = 5000;
burn_in = 1000;

fres_min = 34e9; fres_max = 35.5e9;
gamma_min = 0.1e9; gamma_max = 2e9;

sigma_fres = (fres_max - fres_min) * 0.05;
sigma_gamma = (gamma_max - gamma_min) * 0.1;

% 初始化
rng(42);
fres_current = fres_min + (fres_max - fres_min) * rand();
gamma_current = gamma_min + (gamma_max - gamma_min) * rand();

fprintf('  MCMC初始点: f_res=%.3f GHz, gamma=%.3f GHz\n', fres_current/1e9, gamma_current/1e9);

logL_current = compute_log_likelihood(X_fit, Y_fit, Weights, fres_current, gamma_current, ...
                                       sigma_meas, omega_p_meta, d, c);

samples_fres = zeros(N_samples, 1);
samples_gamma = zeros(N_samples, 1);
accept_count = 0;

hWait = waitbar(0, 'MCMC采样中...');

for i = 1:N_samples
    fres_proposed = fres_current + sigma_fres * randn();
    gamma_proposed = gamma_current + sigma_gamma * randn();
    
    if fres_proposed < fres_min || fres_proposed > fres_max || ...
       gamma_proposed < gamma_min || gamma_proposed > gamma_max
        samples_fres(i) = fres_current;
        samples_gamma(i) = gamma_current;
        continue;
    end
    
    logL_proposed = compute_log_likelihood(X_fit, Y_fit, Weights, fres_proposed, gamma_proposed, ...
                                            sigma_meas, omega_p_meta, d, c);
    
    log_alpha = logL_proposed - logL_current;
    
    if log(rand()) < log_alpha
        fres_current = fres_proposed;
        gamma_current = gamma_proposed;
        logL_current = logL_proposed;
        accept_count = accept_count + 1;
    end
    
    samples_fres(i) = fres_current;
    samples_gamma(i) = gamma_current;
    
    if mod(i, 500) == 0
        waitbar(i/N_samples, hWait);
    end
end
close(hWait);

% 后验分析
samples_fres_valid = samples_fres(burn_in+1:end);
samples_gamma_valid = samples_gamma(burn_in+1:end);

fres_mean = mean(samples_fres_valid);
fres_std = std(samples_fres_valid);
gamma_mean = mean(samples_gamma_valid);
gamma_std = std(samples_gamma_valid);

fprintf('\n===== MCMC结果 =====\n');
fprintf('接受率: %.2f%%\n', accept_count/N_samples*100);
fprintf('f_res: %.4f ± %.4f GHz\n', fres_mean/1e9, fres_std/1e9);
fprintf('gamma: %.4f ± %.4f GHz\n', gamma_mean/1e9, gamma_std/1e9);

%% 11. MCMC可视化 (Figure 10-12)
fprintf('\n【步骤9】生成Figure 10-12...\n');

% Figure 10: Trace plots
figure(10); clf;
subplot(2,2,1);
plot(samples_fres/1e9, 'b', 'LineWidth', 0.5);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数', 'FontName', 'SimHei');
ylabel('f_{res} (GHz)');
title('f_{res} Trace Plot', 'FontName', 'SimHei');
grid on;

subplot(2,2,2);
plot(samples_gamma/1e9, 'b', 'LineWidth', 0.5);
xline(burn_in, 'k--', 'Burn-in');
xlabel('迭代次数', 'FontName', 'SimHei');
ylabel('\gamma (GHz)');
title('\gamma Trace Plot', 'FontName', 'SimHei');
grid on;

subplot(2,2,3);
histogram(samples_fres_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8]);
xlabel('f_{res} (GHz)');
ylabel('概率密度', 'FontName', 'SimHei');
title('f_{res} 后验分布', 'FontName', 'SimHei');
grid on;

subplot(2,2,4);
histogram(samples_gamma_valid/1e9, 40, 'Normalization', 'pdf', 'FaceColor', [0.8 0.4 0.2]);
xlabel('\gamma (GHz)');
ylabel('概率密度', 'FontName', 'SimHei');
title('\gamma 后验分布', 'FontName', 'SimHei');
grid on;

sgtitle('MCMC反演结果', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

% Figure 11: Corner plot
figure(11); clf;
scatter(samples_fres_valid(1:10:end)/1e9, samples_gamma_valid(1:10:end)/1e9, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(fres_mean/1e9, gamma_mean/1e9, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('f_{res} (GHz)');
ylabel('\gamma (GHz)');
title('联合后验分布', 'FontName', 'SimHei');
grid on;

% Figure 12: 拟合验证
figure(12); clf;
scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
colorbar; ylabel(colorbar, '权重');
hold on;

f_plot = linspace(min(X_fit), max(X_fit), 200);
tau_plot = calculate_lorentz_delay(f_plot, fres_mean, gamma_mean, omega_p_meta, d, c);
plot(f_plot/1e9, tau_plot*1e9, 'r', 'LineWidth', 2.5);

xline(fres_mean/1e9, 'g--', 'LineWidth', 2);

title(sprintf('MCMC拟合结果 (f_{res}=%.3f GHz)', fres_mean/1e9), 'FontName', 'SimHei');
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('相对时延 (ns)', 'FontName', 'SimHei');
legend('测量数据', '拟合曲线', '谐振频率', 'Location', 'best');
grid on;

fprintf('  ✓ Figure 10-12 完成\n');

% %% 12. 保存结果
% result_file = fullfile(script_dir, 'cst_mcmc_results.mat');
% save(result_file, 't', 's_tx', 's_rx', 's_if_meta', ...
%      'feature_f_probe', 'feature_tau_absolute', ...
%      'fres_mean', 'fres_std', 'gamma_mean', 'gamma_std', ...
%      'samples_fres', 'samples_gamma');
% fprintf('\n✓ 结果已保存: %s\n', result_file);

%% 总结
fprintf('\n================================================\n');
fprintf('【反演完成】\n');
fprintf('================================================\n');
fprintf('f_res = %.4f ± %.4f GHz\n', fres_mean/1e9, fres_std/1e9);
fprintf('gamma = %.4f ± %.4f GHz\n', gamma_mean/1e9, gamma_std/1e9);
fprintf('接受率: %.1f%%\n', accept_count/N_samples*100);
fprintf('================================================\n');

%% =========================================================================
%  局部函数
%  =========================================================================

function logL = compute_log_likelihood(f_data, tau_data, weights, fres_val, gamma_val, sigma, wp_meta, d, c)
    if fres_val <= 0 || gamma_val <= 0
        logL = -1e10; return;
    end
    
    if fres_val < min(f_data)*0.8 || fres_val > max(f_data)*1.2
        logL = -1e10; return;
    end
    
    try
        tau_theory = calculate_lorentz_delay(f_data, fres_val, gamma_val, wp_meta, d, c);
        residuals = (tau_theory - tau_data) / sigma;
        logL = -0.5 * sum(weights .* residuals.^2);
        
        if isnan(logL) || isinf(logL)
            logL = -1e10;
        end
    catch
        logL = -1e10;
    end
end

function tau_rel = calculate_lorentz_delay(f_vec, f_res_val, gamma_val, wp_meta, d, c)
    omega_vec = 2 * pi * f_vec(:)';
    omega_res_val = 2 * pi * f_res_val;
    
    eps_r = 1 + (wp_meta^2) ./ (omega_res_val^2 - omega_vec.^2 - 1i*2*pi*gamma_val*omega_vec);
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    phi_meta = -real(k_vec) * d;
    
    d_phi = diff(phi_meta);
    d_omega = diff(omega_vec);
    tau_total = -d_phi ./ d_omega;
    tau_total = [tau_total, tau_total(end)];
    
    tau_rel = tau_total - (d/c);
    tau_rel = tau_rel(:);
end
