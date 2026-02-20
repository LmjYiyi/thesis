%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频信号时延提取 - 滑动窗口 ESPRIT
% 参考 LM_MCMC.m 第7节的处理方法
% 输入: hunpin_time_v.txt (ADS仿真已混频信号)
% 输出: 时延-探测频率 散点图
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
%  2. LFMCW 参数设置 (参考 LM_MCMC.m)
% =======================================================================
f_start = 34.4e9;           % 起始频率 (Hz)
f_end   = 37.6e9;           % 终止频率 (Hz)
B       = f_end - f_start;  % 扫频带宽 3.2 GHz

% 关键假设：ADS仿真时长 = 一个完整扫频周期
% (ADS在 ~550 ns 内完成 34.4-37.6 GHz 扫频)
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
% ADS原始混频信号包含:
%   - 和频分量 ~2*fc ≈ 71 GHz (需滤除)
%   - 差频分量 = K*tau ≈ 数十 MHz (目标提取)
%
% 策略: 先粗降采样(简单抽取), 再低通滤波
%        和频混叠到高频, LPF 会自动滤除

% --- Stage 1: 强制均匀重采样 (解决 ADS 变步长问题) ---
fs_dec = 4e9; % 4 GHz 均匀网格
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');

fprintf('\n重采样 Stage1: 均匀采样率=%.1f GHz, 点数=%d\n', ...
    fs_dec/1e9, length(v_dec));

% --- Stage 2: 低通滤波提取差频信号 ---
fc_lp = 200e6;  % 截止频率 200 MHz (远高于预期差频, 远低于混叠和频)
Wn = fc_lp / (fs_dec / 2);  % 归一化截止频率 ≈ 0.1
[b_lp, a_lp] = butter(4, Wn);
s_if = filtfilt(b_lp, a_lp, v_dec);

fprintf('低通滤波完成 (截止 %.0f MHz, Wn=%.3f)\n', fc_lp/1e6, Wn);

% --- Stage 3: 进一步降采样 (4 GHz → 2 GHz) ---
dec2 = 2;
s_proc  = s_if(1:dec2:end);
t_proc  = t_dec(1:dec2:end);
f_s_proc = fs_dec / dec2;   % ~2 GHz
N_proc  = length(s_proc);

fprintf('降采样 Stage2: 因子=%d, 处理采样率=%.1f GHz, 处理点数=%d\n', ...
    dec2, f_s_proc/1e9, N_proc);

%% ======================================================================
%  4. (可选) 差频信号可视化 - 验证滤波效果
% =======================================================================
figure('Color', 'w', 'Position', [50, 500, 1000, 350]);
plot(t_proc * 1e9, s_proc * 1e3, 'b', 'LineWidth', 0.8);
xlabel('时间 (ns)'); ylabel('幅值 (mV)');
title('低通滤波后差频 (IF) 信号'); grid on; axis tight;

%% ======================================================================
%  5. 滑动窗口 + MDL + ESPRIT 时频特征提取
%     (参考 LM_MCMC.m 第7节)
% =======================================================================
fprintf('\n开始滑动窗口 ESPRIT 特征提取...\n');

% 窗口参数
win_frac = 0.03;                          % 窗口占扫频周期的比例
win_len  = max(round(win_frac * N_proc), 64);
step_len = max(round(win_len / 8), 1);    % 步进 = 窗口/8
L_sub    = round(win_len / 2);            % Hankel 矩阵行数

% 存储提取结果
feature_f_probe  = [];
feature_tau      = [];
feature_amplitude = [];

% 信号能量阈值 (跳过无信号区域)
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

    % 窗口中心时间
    t_center = t_proc(idx_start + round(win_len/2));

    % 时间 → 探测频率映射
    f_current_probe = f_start + K * t_center;

    % 避开扫频边缘 (前后各 5%)
    if t_center > 0.95 * T_m || t_center < 0.05 * T_m
        continue;
    end

    % 信号能量检查 - 跳过太弱的窗口 (开头零信号区)
    if rms(x_window) < rms_threshold
        continue;
    end

    % ---- 构建 Hankel 矩阵 ----
    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end

    % ---- 自相关 + 前后向平滑 ----
    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;

    % ---- 特征分解 ----
    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);

    % ---- MDL 准则估计信源数 ----
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

    % ---- TLS-ESPRIT ----
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ ...
          (Us(1:end-1, :)' * Us(2:end, :));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));

    % ---- 频率筛选 ----
    valid_mask = (est_freqs > 50e3) & (est_freqs < f_s_proc/4);
    valid_freqs = est_freqs(valid_mask);

    if isempty(valid_freqs), continue; end

    % 选取最小有效频率 (直达波假设)
    [f_beat_est, ~] = min(valid_freqs);

    % 幅度估计 (加权用)
    amp_est = rms(x_window);

    % 差频 → 时延转换: tau = f_beat / K
    tau_est = f_beat_est / K;

    % 存储结果
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
%  6. 绘制时延散点图及对比理论延迟
% =======================================================================
figure('Color', 'w', 'Position', [100, 100, 900, 600]);

% 加载 ADS 延迟参考数据 (delay.txt)
fprintf('加载 ADS 理论群延迟数据...\n');
try
    delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
    ref_freq = delay_data(:, 1) / 1e9; % 转为 GHz
    ref_tau = delay_data(:, 2) * 1e9;  % 转为 ns
    has_ref = true;
catch
    warning('未找到或无法读取 delay.txt，将跳过参考曲线的绘制。');
    has_ref = false;
end

hold on;

% 绘制 ADS 理论延迟曲线
if has_ref
    plot(ref_freq, ref_tau, 'r-', 'LineWidth', 2, 'DisplayName', 'ADS S参数延迟 (delay.txt)');
end

% 过滤异常点 (排除极端时延)
if ~isempty(feature_tau)
    tau_median = median(feature_tau);
    tau_mad    = median(abs(feature_tau - tau_median));
    valid_plot = abs(feature_tau - tau_median) < 5 * tau_mad;

    scatter(feature_f_probe(valid_plot) / 1e9, ...
            feature_tau(valid_plot) * 1e9, ...
            40, feature_amplitude(valid_plot), 'filled', ...
            'MarkerFaceAlpha', 0.8, 'DisplayName', 'ESPRIT 提取时延');
    
    cb = colorbar; ylabel(cb, '信号幅度 (RMS)');
    
    % 如果有参考数据，限制 Y 轴显示范围以更好对比
    if has_ref
        % 寻找在扫频范围内的参考延迟，根据它设定 y 轴范围
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
    xlabel('探测频率 (GHz)', 'FontSize', 12);
    ylabel('时延 \tau (ns)', 'FontSize', 12);
    title({'ADS混频信号 - ESPRIT提取与S参数延迟对比', ...
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
if ~isempty(feature_tau)
    fprintf('时延中位数:  %.3f ns\n', median(feature_tau)*1e9);
    fprintf('时延范围:    [%.3f, %.3f] ns\n', min(feature_tau)*1e9, max(feature_tau)*1e9);
    fprintf('频率覆盖:    [%.2f, %.2f] GHz\n', ...
        min(feature_f_probe)/1e9, max(feature_f_probe)/1e9);
end
