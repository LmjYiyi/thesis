%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频信号时延提取 - 滑动窗口 ESPRIT (带基准平移校准)
% 输入: hunpin_time_v.txt (ADS仿真已混频信号)
% 输出: 时延-探测频率 散点图 (经非色散区平移校准)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% ======================================================================
%  1. 加载 ADS 混频信号数据
% =======================================================================
fprintf('正在加载ADS混频信号数据...\n');
data = readmatrix('hunpin_time_v.txt', 'FileType', 'text', 'NumHeaderLines', 1);
t_raw = data(:, 1);   % 时间 (s)
v_raw = data(:, 2);   % 电压 (V)

% 去除末尾 NaN / 空行
valid = ~isnan(t_raw) & ~isnan(v_raw);
t_raw = t_raw(valid);
v_raw = v_raw(valid);
N_total = length(t_raw);

% 确定采样参数 (跳过开头非均匀采样区间)
dt_typical = median(diff(t_raw(1000:2000)));
f_s_original = 1 / dt_typical;
T_data = t_raw(end) - t_raw(1);

fprintf('  数据点数: %d\n', N_total);
fprintf('  采样率:   %.2f THz\n', f_s_original / 1e12);
fprintf('  数据时长: %.2f ns\n', T_data * 1e9);

%% ======================================================================
%  2. LFMCW 参数设置
% =======================================================================
f_start = 34.4e9;           % 起始频率 (Hz)
f_end   = 37.6e9;           % 终止频率 (Hz)
B       = f_end - f_start;  % 扫频带宽 3.2 GHz

T_m = T_data;
K   = B / T_m;              % 调频斜率 (Hz/s)

fprintf('\nLFMCW 参数:\n');
fprintf('  f_start = %.2f GHz\n', f_start / 1e9);
fprintf('  f_end   = %.2f GHz\n', f_end / 1e9);
fprintf('  B       = %.2f GHz\n', B / 1e9);
fprintf('  T_m     = %.2f ns\n', T_m * 1e9);
fprintf('  K       = %.3e Hz/s\n', K);

%% ======================================================================
%  3. 信号预处理：降采样 + 低通滤波 提取差频
% =======================================================================
% --- Stage 1: 强制均匀重采样 (解决 ADS 变步长问题) ---
fs_dec = 4e9; % 4 GHz 均匀网格
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');

fprintf('\n重采样 Stage1: 均匀采样率=%.1f GHz, 点数=%d\n', fs_dec/1e9, length(v_dec));

% --- Stage 2: 低通滤波提取差频信号 ---
fc_lp = 200e6;  % 截止频率 200 MHz
Wn = fc_lp / (fs_dec / 2);  
[b_lp, a_lp] = butter(4, Wn);
s_if = filtfilt(b_lp, a_lp, v_dec);

fprintf('低通滤波完成 (截止 %.0f MHz, Wn=%.3f)\n', fc_lp/1e6, Wn);

% --- Stage 3: 进一步降采样 (4 GHz → 2 GHz) ---
dec2 = 2;
s_proc  = s_if(1:dec2:end);
t_proc  = t_dec(1:dec2:end);
f_s_proc = fs_dec / dec2;   
N_proc  = length(s_proc);

fprintf('降采样 Stage2: 因子=%d, 处理采样率=%.1f GHz, 处理点数=%d\n', dec2, f_s_proc/1e9, N_proc);

%% ======================================================================
%  4. 滑动窗口 + MDL + ESPRIT 时频特征提取
% =======================================================================
fprintf('\n开始滑动窗口 ESPRIT 特征提取...\n');

win_frac = 0.03;                          
win_len  = max(round(win_frac * N_proc), 64);
step_len = max(round(win_len / 8), 1);    
L_sub    = round(win_len / 2);            

feature_f_probe  = [];
feature_tau      = [];
feature_amplitude = [];

rms_threshold = max(abs(s_proc)) * 0.005;

num_windows = floor((N_proc - win_len) / step_len) + 1;
fprintf('  窗口长度: %d 点 (%.1f ns)\n', win_len, win_len/f_s_proc*1e9);
fprintf('  步进长度: %d 点\n', step_len);
fprintf('  总窗口数: %d\n', num_windows);

hWait = waitbar(0, 'ESPRIT 特征提取中...');

for i = 1:num_windows
    idx_start = (i-1) * step_len + 1;
    idx_end   = idx_start + win_len - 1;
    if idx_end > N_proc, break; end

    x_window = s_proc(idx_start:idx_end);
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;

    if t_center > 0.95 * T_m || t_center < 0.05 * T_m
        continue;
    end

    if rms(x_window) < rms_threshold
        continue;
    end

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
        noise_evals(noise_evals < 1e-30) = 1e-30;
        g_mean = prod(noise_evals)^(1/length(noise_evals));
        a_mean = mean(noise_evals);
        term1 = -(p - k_mdl) * N_snaps * log(g_mean / a_mean);
        term2 = 0.5 * k_mdl * (2*p - k_mdl) * log(N_snaps);
        mdl_cost(k_mdl+1) = term1 + term2;
    end
    [~, min_idx] = min(mdl_cost);
    k_est = min_idx - 1;

    num_sources = max(1, k_est);
    num_sources = min(num_sources, 3);

    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ ...
          (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));

    valid_mask = (est_freqs > 50e3) & (est_freqs < f_s_proc/4);
    valid_freqs = est_freqs(valid_mask);

    if isempty(valid_freqs), continue; end

    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;

    feature_f_probe   = [feature_f_probe,   f_current_probe];
    feature_tau       = [feature_tau,       tau_est];
    feature_amplitude = [feature_amplitude, amp_est];

    if mod(i, 20) == 0
        waitbar(i / num_windows, hWait);
    end
end
close(hWait);

fprintf('特征提取完成: 共 %d 个有效数据点\n', length(feature_tau));

%% ======================================================================
%  5. 系统基准平移校准 (Baseline Subtraction)
% =======================================================================
fprintf('\n执行系统基准平移校准...\n');

% 定义非色散区域（平坦区）的频率范围，以此作为系统固有延迟的参考基底
calib_f_min = 35.5e9;
calib_f_max = 36.2e9;

idx_calib = (feature_f_probe >= calib_f_min) & (feature_f_probe <= calib_f_max);

if sum(idx_calib) > 0
    baseline_delay = mean(feature_tau(idx_calib));
    fprintf('找到 %d 个平坦区参考点，计算系统基准延迟为: %.3f ns\n', sum(idx_calib), baseline_delay * 1e9);
    feature_tau_calibrated = feature_tau - baseline_delay;
else
    warning('未在平坦区域 (%.1f - %.1f GHz) 找到有效特征点，将跳过平移校准。', calib_f_min/1e9, calib_f_max/1e9);
    feature_tau_calibrated = feature_tau; 
    baseline_delay = 0;
end

%% ======================================================================
%  6. 绘制校准后的时延散点图及理论对比
% =======================================================================
figure('Color', 'w', 'Position', [100, 100, 950, 600]);

fprintf('加载 ADS 理论群延迟数据...\n');
try
    delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
    ref_freq = delay_data(:, 1) / 1e9; 
    ref_tau = delay_data(:, 2) * 1e9;  
    has_ref = true;
catch
    warning('未找到 delay.txt，将跳过参考曲线的绘制。');
    has_ref = false;
end

hold on;

if has_ref
    plot(ref_freq, ref_tau, 'r-', 'LineWidth', 2, 'DisplayName', 'ADS S参数理论延迟 (仅滤波器)');
end

if ~isempty(feature_tau_calibrated)
    % 使用校准后的数据进行异常值过滤
    tau_median = median(feature_tau_calibrated);
    tau_mad    = median(abs(feature_tau_calibrated - tau_median));
    valid_plot = abs(feature_tau_calibrated - tau_median) < 5 * tau_mad;

    scatter(feature_f_probe(valid_plot) / 1e9, ...
            feature_tau_calibrated(valid_plot) * 1e9, ...
            45, feature_amplitude(valid_plot), 'filled', ...
            'MarkerFaceAlpha', 0.85, ...
            'DisplayName', sprintf('ESPRIT 提取时延 (已减去基准 %.2fns)', baseline_delay*1e9));
    
    cb = colorbar; ylabel(cb, '信号幅度 (RMS)');
    
    if has_ref
        idx_band = ref_freq >= f_start/1e9 & ref_freq <= f_end/1e9;
        if any(idx_band)
            min_tau = min(ref_tau(idx_band));
            max_tau = max(ref_tau(idx_band));
            padding = (max_tau - min_tau) * 0.5;
            if padding == 0, padding = 1; end
            ylim([min_tau - padding, max_tau + padding]);
        end
    end

    grid on;
    set(gca, 'GridAlpha', 0.3);
    xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('纯滤波器时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
    title({'ADS混频信号 - 基准平移校准后的ESPRIT时延对比', ...
           sprintf('有效点数: %d / %d', sum(valid_plot), length(feature_tau))}, ...
           'FontSize', 14);
    xlim([f_start/1e9, f_end/1e9]);
    legend('Location', 'best', 'FontSize', 11);
else
    text(0.5, 0.5, '未提取到有效数据点，请检查参数设置', ...
        'HorizontalAlignment', 'center', 'FontSize', 14);
    title('ADS混频信号 - ESPRIT时延提取 (无有效结果)');
end

fprintf('\n===== 处理完毕 =====\n');
if ~isempty(feature_tau_calibrated)
    fprintf('校准后时延中位数:  %.3f ns\n', median(feature_tau_calibrated)*1e9);
    fprintf('频率覆盖:          [%.2f, %.2f] GHz\n', min(feature_f_probe)/1e9, max(feature_f_probe)/1e9);
end