%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时域数据时延轨迹提取（滑动窗口 MDL-ESPRIT）
% 输入: ex/lowpassfilter_filter.csv (有滤波器, 已100MHz低通, 示波器采集)
% LFMCW 参数: 34–37 GHz, T_m = 50 μs
% 流程: 周期叠加平均 → 降采样 → 滑动窗口 ESPRIT → 时延轨迹
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

%% 1. 实验 LFMCW 参数
f_start = 34e9;         % 起始频率 (Hz)
f_end   = 37e9;         % 终止频率 (Hz)
B       = f_end - f_start;   % 3 GHz
T_m     = 50e-6;              % 扫频周期 50 μs
K       = B / T_m;            % 6e13 Hz/s

fprintf('===== 实测时延轨迹提取 =====\n');
fprintf('LFMCW: %.0f–%.0f GHz, B=%.1f GHz, T_m=%.0f μs, K=%.2e Hz/s\n', ...
    f_start/1e9, f_end/1e9, B/1e9, T_m*1e6, K);

%% 2. 加载数据
fprintf('\n正在加载 lowpassfilter_filter.csv ...\n');
data = readmatrix(fullfile(script_dir, 'data', 'lowpassfilter_filter.csv'));
t_raw = data(:, 1);
v_raw = data(:, 2);

dt = median(diff(t_raw));
fs = round(1 / dt);          % 采样率 ~200 MHz
N_total = length(t_raw);

fprintf('  采样率: %.0f MHz, 总点数: %d, 时长: %.2f ms\n', ...
    fs/1e6, N_total, (t_raw(end) - t_raw(1)) * 1e3);

%% 3. 去直流 + 分段叠加平均
v_raw = v_raw - mean(v_raw);

N_per   = round(T_m * fs);           % 每周期点数 (10000)
N_sweep = floor(N_total / N_per);    % 完整周期数 (100)

v_mat = reshape(v_raw(1 : N_sweep * N_per), N_per, N_sweep);
v_avg = mean(v_mat, 2);             % 叠加平均后的单周期波形

fprintf('  每周期 %d 点, %d 个完整周期, 叠加平均 SNR+%.1f dB\n', ...
    N_per, N_sweep, 10*log10(N_sweep));

%% 4. 降采样至 20 MHz + 带通滤波
% beat freq ~60-150 kHz, 20 MHz 仍远超 Nyquist, 但保留更多点数供 ESPRIT 使用
ds = 10;
v_ds    = v_avg(1:ds:end);
fs_proc = fs / ds;                   % 20 MHz
N_ds    = length(v_ds);

% 高通滤波：去除 DC 残留和极低频漂移，保留 beat 频率 (~60-150 kHz)
% 用高通替代带通，避免窄带 butter 数值不稳定
f_hp = 10e3 / (fs_proc / 2);        % 10 kHz 高通截止
[b_hp, a_hp] = butter(2, f_hp, 'high');
[b_hp, a_hp] = butter(2, f_hp, 'high');
v_proc  = filtfilt(b_hp, a_hp, v_ds);

N_proc  = length(v_proc);
t_proc  = (0:N_proc-1).' / fs_proc;

fprintf('  降采样 ×%d → fs_proc=%.0f MHz, N_proc=%d\n', ds, fs_proc/1e6, N_proc);
fprintf('  高通滤波: >%.0f kHz (去DC)\n', 10);

%% 5. 滑动窗口 ESPRIT + MDL
% --- 窗口参数 ---
% win_len ~150 @ 20 MHz → 7.5 μs → 频率跨度 ~450 MHz
% 比原版 10 μs 稍短, 兼顾边缘分辨与通带估计精度
win_len  = max(round(N_proc * 0.15), 80);  % ~150 samples, 下限 80
step_len = max(round(win_len / 12), 1);    % 步进适中
L_sub    = round(win_len / 2);
rms_thr  = max(abs(v_proc)) * 0.01;
num_win  = floor((N_proc - win_len) / step_len) + 1;

f_probe_arr = zeros(num_win, 1);
tau_arr     = zeros(num_win, 1);
amp_arr     = zeros(num_win, 1);
cnt = 0;

fprintf('\nESPRIT: win=%d (%.1f μs), step=%d, L_sub=%d, 共 %d 窗口\n', ...
    win_len, win_len/fs_proc*1e6, step_len, L_sub, num_win);

for i = 1:num_win
    idx = (i-1)*step_len + 1 : (i-1)*step_len + win_len;
    if idx(end) > N_proc, break; end

    x_win = v_proc(idx);
    t_c   = t_proc(idx(round(win_len/2)));

    % RMS 门限：跳过噪声段（探测频率不在滤波器通带内时无信号）
    if rms(x_win) < rms_thr
        fprintf('  [跳过] 窗口 %3d: t_c=%6.2f μs — RMS=%.6f < 门限 %.6f\n', ...
            i, t_c*1e6, rms(x_win), rms_thr);
        continue;
    end

    % --- 构建 Hankel 矩阵 ---
    M_sub = win_len - L_sub + 1;
    X_h = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_h(:, k) = x_win(k : k+L_sub-1).';
    end

    % --- 前后向空间平滑 ---
    R_fwd = (X_h * X_h') / M_sub;
    J     = fliplr(eye(L_sub));
    R_x   = (R_fwd + J * conj(R_fwd) * J) / 2;

    [V, D] = eig(R_x);
    [lam, id] = sort(diag(D), 'descend');
    V = V(:, id);

    % --- MDL 信源估计 ---
    mdl_v = zeros(length(lam), 1);
    for kk = 0:length(lam)-1
        ns = lam(kk+1:end);
        ns(ns < 1e-30) = 1e-30;
        mdl_v(kk+1) = -(length(lam)-kk) * M_sub * ...
            log(prod(ns)^(1/length(ns)) / mean(ns)) + ...
            0.5 * kk * (2*length(lam)-kk) * log(M_sub);
    end
    [~, k_est] = min(mdl_v);
    num_s = min(max(1, k_est-1), 3);

    % --- ESPRIT 旋转不变估计 ---
    Us  = V(:, 1:num_s);
    Phi = (Us(1:end-1,:)' * Us(1:end-1,:)) \ (Us(1:end-1,:)' * Us(2:end,:));
    eig_vals = eig(Phi);
    est_f = abs(angle(eig_vals)) * fs_proc / (2*pi);

    % 保留 20 kHz – fs/4 范围内的频率分量（与高通截止匹配）
    valid_mask = est_f > 20e3 & est_f < fs_proc/4;
    est_f = est_f(valid_mask);
    if isempty(est_f)
        fprintf('  [跳过] 窗口 %3d: t_c=%6.2f μs — 无有效频率分量 (20kHz–fs/4)\n', i, t_c*1e6);
        continue;
    end

    % 选取主分量：投影幅度排序 + 频率合理性检查
    % 物理上界: tau_max ~5 ns → f_beat_max = K * 5e-9 = 300 kHz
    f_beat_max = 300e3;

    % 计算各分量投影能量并按强度排序
    proj_power = zeros(numel(est_f), 1);
    for jj = 1:numel(est_f)
        steering = exp(1j * 2*pi * est_f(jj) / fs_proc * (0:L_sub-1).');
        proj_power(jj) = abs(steering' * V(:,1))^2;
    end
    [~, rank_idx] = sort(proj_power, 'descend');

    % 按投影强度依次尝试，取第一个频率合理的分量
    f_beat = [];
    for jj = rank_idx.'
        if est_f(jj) <= f_beat_max
            f_beat = est_f(jj);
            break;
        end
    end
    if isempty(f_beat)
        fprintf('  [跳过] 窗口 %3d: t_c=%6.2f μs — 所有分量 >%.0f kHz (est_f=[%s] kHz)\n', ...
            i, t_c*1e6, f_beat_max/1e3, strjoin(compose('%.1f', est_f/1e3), ', '));
        continue;
    end

    cnt = cnt + 1;
    f_probe_arr(cnt) = f_start + K * t_c;
    tau_arr(cnt)     = f_beat / K;
    amp_arr(cnt)     = rms(x_win);

    % 逐窗口详细记录
    fprintf('  [保留] 窗口 %3d: t_c=%6.2f μs, f_probe=%7.3f GHz, f_beat=%6.1f kHz, tau=%5.2f ns, MDL→%d源, rms=%.5f\n', ...
        i, t_c*1e6, f_probe_arr(cnt)/1e9, f_beat/1e3, tau_arr(cnt)*1e9, num_s, amp_arr(cnt));
end

f_probe_arr = f_probe_arr(1:cnt);
tau_arr     = tau_arr(1:cnt);
amp_arr     = amp_arr(1:cnt);

fprintf('\n  ESPRIT 原始散点: %d / %d 窗口 (跳过 %d)\n', cnt, num_win, num_win - cnt);

%% 5b. 散点后处理：幅度门限 + IQR 异常剔除
fprintf('\n----- 散点后处理 -----\n');

% --- 幅度归一化门限 ---
amp_norm = amp_arr / (max(amp_arr) + eps);
mask_amp = amp_norm > 0.15;
fprintf('【幅度门限】 阈值=0.15, 剔除 %d 个:\n', sum(~mask_amp));
for pp = find(~mask_amp).'
    fprintf('    #%d: f=%.3f GHz, tau=%.2f ns, amp_norm=%.3f\n', ...
        pp, f_probe_arr(pp)/1e9, tau_arr(pp)*1e9, amp_norm(pp));
end

% --- IQR 异常剔除（针对时延） ---
tau_masked = tau_arr(mask_amp);
tau_q25 = prctile(tau_masked, 25);
tau_q75 = prctile(tau_masked, 75);
tau_iqr = tau_q75 - tau_q25;
tau_lo  = tau_q25 - 2.0 * tau_iqr;
tau_hi  = tau_q75 + 2.0 * tau_iqr;
mask_iqr = tau_arr >= tau_lo & tau_arr <= tau_hi;
fprintf('【IQR剔除】 Q25=%.2f ns, Q75=%.2f ns, IQR=%.2f ns, 允许=[%.2f, %.2f] ns, 剔除 %d 个:\n', ...
    tau_q25*1e9, tau_q75*1e9, tau_iqr*1e9, tau_lo*1e9, tau_hi*1e9, sum(mask_amp & ~mask_iqr));
for pp = find(mask_amp & ~mask_iqr).'
    fprintf('    #%d: f=%.3f GHz, tau=%.2f ns\n', pp, f_probe_arr(pp)/1e9, tau_arr(pp)*1e9);
end

% --- 移动中值平滑剔除局部跳点 ---
[f_sorted, sort_idx] = sort(f_probe_arr);
tau_sorted = tau_arr(sort_idx);
local_span = max(5, 2 * floor(cnt / 40) + 1);
if mod(local_span, 2) == 0, local_span = local_span + 1; end
tau_med = movmedian(tau_sorted, local_span);
tau_dev = abs(tau_sorted - tau_med);
dev_thr = max(3 * 1.4826 * movmedian(tau_dev, local_span), 0.3e-9);
mask_local = true(cnt, 1);
mask_local(sort_idx) = tau_dev <= dev_thr;
n_local_reject = sum(mask_amp & mask_iqr & ~mask_local);
fprintf('【局部连续性】 移动中值 span=%d, 剔除 %d 个:\n', local_span, n_local_reject);
for pp = find(mask_amp & mask_iqr & ~mask_local).'
    si_pp = find(sort_idx == pp);
    fprintf('    #%d: f=%.3f GHz, tau=%.2f ns, 中值=%.2f ns, 偏差=%.2f ns\n', ...
        pp, f_probe_arr(pp)/1e9, tau_arr(pp)*1e9, tau_med(si_pp)*1e9, tau_dev(si_pp)*1e9);
end

% --- 综合掩码 ---
mask_clean = mask_amp & mask_iqr & mask_local;
fprintf('\n  汇总: 原始 %d → 幅度 %d → +IQR %d → +连续性 %d\n', ...
    cnt, sum(mask_amp), sum(mask_amp & mask_iqr), sum(mask_clean));

f_probe_arr = f_probe_arr(mask_clean);
tau_arr     = tau_arr(mask_clean);
amp_arr     = amp_arr(mask_clean);

%% 5c. 频率轴校准：双锚点线性映射
% 单纯平移不足以校正频率轴 —— K 或 t_offset 均存在偏差
% 方法：在散点左/右半段分别找时延峰值，将其映射到已知通带边缘
f_edge_lo = 36.5e9;    % 滤波器通带下边缘 (Hz)
f_edge_hi = 37.5e9;    % 滤波器通带上边缘 (Hz)

% 按频率排序，优先在左右边缘区域寻找锚点，避免右锚点选得过早
[f_cal_sorted, cal_si] = sort(f_probe_arr);
tau_cal_sorted = tau_arr(cal_si);
N_half = round(numel(f_probe_arr) / 2);

% 左半段：取时延最大的点对应的频率（用局部平滑避免单点噪声）
tau_left_smooth = movmean(tau_cal_sorted(1:N_half), max(3, round(N_half/5)));
[~, idx_lo] = max(tau_left_smooth);
f_anchor_lo = f_cal_sorted(idx_lo);

% 右半段
tau_right_smooth = movmean(tau_cal_sorted(N_half+1:end), max(3, round((numel(f_probe_arr)-N_half)/5)));
[~, idx_hi_rel] = max(tau_right_smooth);
f_anchor_hi = f_cal_sorted(N_half + idx_hi_rel);

% 线性映射: f_cal = a * f_raw + b
a_cal = (f_edge_hi - f_edge_lo) / (f_anchor_hi - f_anchor_lo);
b_cal = f_edge_lo - a_cal * f_anchor_lo;
f_probe_arr = a_cal * f_probe_arr + b_cal;

fprintf('\n===== 频率轴校准 (双锚点) =====\n');
fprintf('  左锚点: 原始 %.3f GHz → 映射到 %.1f GHz (通带下边缘)\n', f_anchor_lo/1e9, f_edge_lo/1e9);
fprintf('  右锚点: 原始 %.3f GHz → 映射到 %.1f GHz (通带上边缘)\n', f_anchor_hi/1e9, f_edge_hi/1e9);
fprintf('  线性系数: a=%.4f, b=%.3f GHz\n', a_cal, b_cal/1e9);
fprintf('  等效 K 校正因子: %.4f (原K=%.2e → 有效K=%.2e Hz/s)\n', ...
    a_cal, K, K*a_cal);

N_final = numel(f_probe_arr);
fprintf('\n===== 最终散点 (%d 个, 校准后) =====\n', N_final);
fprintf('  %-4s  %-12s  %-10s  %-10s\n', '序号', 'f_probe(GHz)', 'tau(ns)', 'amp_rms');
for pp = 1:N_final
    fprintf('  %-4d  %-12.3f  %-10.2f  %-10.5f\n', ...
        pp, f_probe_arr(pp)/1e9, tau_arr(pp)*1e9, amp_arr(pp));
end
fprintf('  ————————————————————————————————————\n');
fprintf('  时延范围: %.2f – %.2f ns (中位数 %.2f ns)\n', ...
    min(tau_arr)*1e9, max(tau_arr)*1e9, median(tau_arr)*1e9);
fprintf('  频率范围: %.2f – %.2f GHz\n', ...
    min(f_probe_arr)/1e9, max(f_probe_arr)/1e9);

%% 6. 频率排序（用于绘图）
[f_sort, si] = sort(f_probe_arr);
tau_sort = tau_arr(si);

%% 7. 绘图
figure('Color', 'w', 'Position', [100, 100, 900, 500]);
hold on;
scatter(f_probe_arr/1e9, tau_arr*1e9, 40, [0.55 0.55 0.55], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
hold off;

grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title('实测 LFMCW 时延轨迹', 'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.3);
xlim([36.0, 38.0]);   % 滤波器通带 36.5–37.5 GHz, 留余量
legend({'ESPRIT 散点'}, 'Location', 'northeast', 'FontSize', 11);

export_thesis_figure(gcf, 'exp_delay_trajectory', 14, 300);

fprintf('\n频率轴已根据滤波器通带 %.1f–%.1f GHz 完成双锚点校准。\n', f_edge_lo/1e9, f_edge_hi/1e9);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 本地函数：统一论文插图风格并自动导出
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 300; end

height_cm = width_cm * 0.618;
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures_export');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPosition', [0, 0, width_cm, height_cm], ...
    'PaperSize', [width_cm, height_cm]);

ax_all = findall(fig_handle, 'Type', 'axes');
for i_ax = 1:numel(ax_all)
    set(ax_all(i_ax), ...
        'FontName', 'SimHei', ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.20, ...
        'TickDir', 'out');
end

line_all = findall(fig_handle, 'Type', 'line');
for i_ln = 1:numel(line_all)
    if strcmp(get(line_all(i_ln), 'LineStyle'), 'none')
        set(line_all(i_ln), 'LineWidth', 1.0);
    else
        set(line_all(i_ln), 'LineWidth', 1.5);
    end
end

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
fprintf('【导出】%s\n', file_tiff);
end
