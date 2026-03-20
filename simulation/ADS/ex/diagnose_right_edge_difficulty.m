%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 右侧边缘提取困难成因诊断
% 用途：逐项验证右侧为何需要特殊处理，输出定量诊断图
% 诊断维度：
%   (a) 窗口边界截断程度
%   (b) 群时延局部斜率
%   (c) 特征值可分离度（子空间质量）
%   (d) 多窗口/多配置一致性
%   (e) 有效信号幅度剖面
%   (f) 候选模态竞争程度
% 依赖文件：esprit_extract.m, trajectory_postprocess.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
esp = esprit_extract();

%% 1. LFMCW 参数与数据加载（同主线脚本）
f_start = 34e9;
f_end   = 37e9;
B       = f_end - f_start;
T_m     = 50e-6;
K       = B / T_m;

data_file = fullfile(script_dir, 'data', 'lowpassfilter_filter.csv');
data = readmatrix(data_file);
t_raw = data(:, 1);
v_raw = data(:, 2);

dt = median(diff(t_raw));
fs = round(1 / dt);
N_total = length(t_raw);

v_raw = v_raw - mean(v_raw);
N_per   = round(T_m * fs);
N_sweep = floor(N_total / N_per);
v_mat   = reshape(v_raw(1:N_sweep * N_per), N_per, N_sweep);
v_avg   = mean(v_mat, 2);

ds = 10;
v_ds    = v_avg(1:ds:end);
fs_proc = fs / ds;

f_hp_cut = 10e3;
[b_hp, a_hp] = butter(2, f_hp_cut / (fs_proc / 2), 'high');
v_proc = filtfilt(b_hp, a_hp, v_ds);
t_proc = (0:length(v_proc)-1).' / fs_proc;
N_proc = length(v_proc);

f_valid_lo = 20e3;
f_beat_max = 300e3;

fprintf('===== 右侧边缘提取困难成因诊断 =====\n');
fprintf('N_proc=%d, fs_proc=%.0f MHz\n', N_proc, fs_proc/1e6);

%% 2. 频率轴校准（复用主线流程，获取 a_cal, b_cal）
pp = trajectory_postprocess();
rms_thr = max(abs(v_proc)) * 0.01;

cfg_cal.win_len  = 150;
cfg_cal.step_len = 13;
cfg_cal.L_sub    = 75;
cfg_cal.name     = '校准用';
base_for_cal = esp.run_fixed(v_proc, t_proc, fs_proc, f_start, K, ...
    rms_thr, cfg_cal, f_valid_lo, f_beat_max, false);
base_clean_cal = pp.clean(base_for_cal, false, cfg_cal.name);
base_cal = pp.calibrate(base_clean_cal, K, true, cfg_cal.name);

a_cal = base_cal.a_cal;
b_cal = base_cal.b_cal;
fprintf('频率轴校准: a_cal=%.4f, b_cal=%.3f GHz\n', a_cal, b_cal/1e9);
fprintf('校准后范围: %.3f - %.3f GHz\n', ...
    min(base_cal.f_probe)/1e9, max(base_cal.f_probe)/1e9);

%% 3. 配置诊断参数
win_len   = 150;
L_sub     = 75;
step_diag = 5;

% 多配置用于一致性测试
multi_wins = [100, 130, 150, 180, 220];
multi_Lsub_ratios = [1/3, 2/5, 1/2];

%% 4. 逐窗口诊断扫描
centers = (round(win_len/2)+1 : step_diag : N_proc-round(win_len/2));
N_diag = numel(centers);

diag_f_probe     = zeros(N_diag, 1);     % 瞬时探测频率
diag_rms         = zeros(N_diag, 1);     % (e) 窗内 RMS 幅度
diag_boundary    = zeros(N_diag, 1);     % (a) 距周期边界的相对距离 [0,1]
diag_eig_gap     = zeros(N_diag, 1);     % (c) 首特征值 / 次特征值比
diag_mdl_margin  = zeros(N_diag, 1);     % (c) MDL 判决余量
diag_n_modes     = zeros(N_diag, 1);     % (f) 有效候选模态数
diag_mode_spread = zeros(N_diag, 1);     % (f) 候选模态时延极差
diag_tau_best    = NaN(N_diag, 1);       % 最强模态时延

fprintf('\n逐窗口诊断扫描: %d 个位置...\n', N_diag);

for ii = 1:N_diag
    ci = centers(ii);
    is = max(1, min(ci - floor(win_len/2), N_proc - win_len + 1));
    idx = is : is + win_len - 1;
    x_win = v_proc(idx);

    % 瞬时探测频率（校准后坐标）
    t_c = t_proc(ci);
    f_raw = f_start + K * t_c;
    diag_f_probe(ii) = a_cal * f_raw + b_cal;

    % (a) 距周期边界的相对距离
    diag_boundary(ii) = min(idx(1) - 1, N_proc - idx(end)) / (N_proc / 2);

    % (e) 窗内 RMS
    diag_rms(ii) = rms(x_win);

    % --- ESPRIT 内部诊断 ---
    M_sub = win_len - L_sub + 1;
    X_h = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_h(:, k) = x_win(k : k + L_sub - 1).';
    end
    R_fwd = (X_h * X_h') / M_sub;
    J = fliplr(eye(L_sub));
    R_x = (R_fwd + J * conj(R_fwd) * J) / 2;
    [V, D] = eig(R_x);
    [lam, id] = sort(diag(D), 'descend');
    V = V(:, id);

    % (c) 特征值可分离度
    lam_pos = lam(lam > 1e-30);
    if numel(lam_pos) >= 2
        diag_eig_gap(ii) = lam_pos(1) / lam_pos(2);
    else
        diag_eig_gap(ii) = NaN;
    end

    % (c) MDL 判决余量
    mdl_v = zeros(length(lam), 1);
    for kk = 0:length(lam)-1
        ns = lam(kk + 1:end);
        ns(ns < 1e-30) = 1e-30;
        mdl_v(kk + 1) = -(length(lam) - kk) * M_sub * ...
            log(prod(ns)^(1 / length(ns)) / mean(ns)) + ...
            0.5 * kk * (2 * length(lam) - kk) * log(M_sub);
    end
    [mdl_min, k_est] = min(mdl_v);
    mdl_sorted = sort(mdl_v);
    if numel(mdl_sorted) >= 2
        diag_mdl_margin(ii) = mdl_sorted(2) - mdl_sorted(1);
    end

    % (f) 候选模态数和模态极差
    [fp, ta, ~, ~, ~] = esp.all_modes( ...
        x_win, idx, t_proc, fs_proc, f_start, K, ...
        rms_thr, L_sub, f_valid_lo, f_beat_max);
    diag_n_modes(ii) = numel(ta);
    if numel(ta) >= 2
        diag_mode_spread(ii) = max(ta) - min(ta);
    end
    if ~isempty(ta)
        [~, best_idx] = max(abs(ta));
        diag_tau_best(ii) = ta(best_idx);
    end
end

%% 5. (b) 群时延局部斜率（基于校准后粗提取结果）
fprintf('\n计算群时延局部斜率...\n');

[f_sort, si] = sort(base_cal.f_probe);
tau_sort = base_cal.tau(si);

% 局部斜率：中心差分
slope_f   = zeros(numel(f_sort), 1);
slope_tau = zeros(numel(f_sort), 1);
hw = 3;
for ii = 1+hw : numel(f_sort)-hw
    df = f_sort(ii+hw) - f_sort(ii-hw);
    dt_val = tau_sort(ii+hw) - tau_sort(ii-hw);
    slope_f(ii)   = f_sort(ii);
    slope_tau(ii)  = abs(dt_val / df);
end
slope_mask = slope_f > 0;
slope_f    = slope_f(slope_mask);
slope_tau  = slope_tau(slope_mask);

%% 6. (d) 多窗口/多配置一致性
fprintf('计算多配置一致性...\n');
step_consist = 15;
centers_consist = (round(max(multi_wins)/2)+1 : step_consist : ...
    N_proc - round(max(multi_wins)/2));
N_consist = numel(centers_consist);

consist_f_probe = zeros(N_consist, 1);
consist_tau_std = zeros(N_consist, 1);
consist_tau_iqr = zeros(N_consist, 1);
consist_n_valid = zeros(N_consist, 1);

for ii = 1:N_consist
    ci = centers_consist(ii);
    t_c = t_proc(ci);
    f_raw = f_start + K * t_c;
    consist_f_probe(ii) = a_cal * f_raw + b_cal;

    tau_candidates = [];
    for iw = 1:numel(multi_wins)
        wl = multi_wins(iw);
        if ci - floor(wl/2) < 1 || ci + floor(wl/2) > N_proc
            continue;
        end
        is = max(1, min(ci - floor(wl/2), N_proc - wl + 1));
        idx = is : is + wl - 1;

        for ir = 1:numel(multi_Lsub_ratios)
            Ls = max(4, round(wl * multi_Lsub_ratios(ir)));

            [ok, ~, tv, ~, ~, ~] = esp.single_window( ...
                v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
                rms_thr, Ls, f_valid_lo, f_beat_max);
            if ok
                tau_candidates(end+1) = tv; %#ok<AGROW>
            end
        end
    end

    consist_n_valid(ii) = numel(tau_candidates);
    if numel(tau_candidates) >= 2
        consist_tau_std(ii) = std(tau_candidates);
        consist_tau_iqr(ii) = iqr(tau_candidates);
    end
end

%% 7. 绘图
fprintf('\n生成诊断图...\n');

f_ghz = diag_f_probe / 1e9;
% 区域标识
f_left_hi  = 36.62;
f_flat_lo  = 36.78;
f_flat_hi  = 37.22;
f_right_lo = 37.38;

figure('Color', 'w', 'Position', [50, 50, 1400, 900]);

% --- (a) 窗口边界截断程度 ---
subplot(3, 2, 1);
plot(f_ghz, diag_boundary, '.', 'Color', [0.16 0.46 0.72], 'MarkerSize', 4);
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
ylabel('边界距离 (归一化)');
title('(a) 窗口边界支撑度');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

% --- (e) 窗内 RMS 幅度 ---
subplot(3, 2, 2);
plot(f_ghz, diag_rms * 1e3, '.', 'Color', [0.16 0.46 0.72], 'MarkerSize', 4);
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
ylabel('RMS (mV)');
title('(e) 窗内有效信号幅度');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

% --- (c) 特征值可分离度 ---
subplot(3, 2, 3);
semilogy(f_ghz, diag_eig_gap, '.', 'Color', [0.16 0.46 0.72], 'MarkerSize', 4);
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
ylabel('\lambda_1 / \lambda_2');
title('(c) 特征值间隔比（子空间可分离度）');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

% --- (b) 群时延局部斜率 ---
subplot(3, 2, 4);
semilogy(slope_f / 1e9, slope_tau * 1e9, '.', 'Color', [0.90 0.50 0.12], 'MarkerSize', 5);
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
ylabel('|d\tau/df| (ns/GHz)');
title('(b) 群时延局部斜率');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

% --- (f) 候选模态数 & 模态竞争 ---
subplot(3, 2, 5);
yyaxis left;
plot(f_ghz, diag_n_modes, '.', 'Color', [0.16 0.46 0.72], 'MarkerSize', 4);
ylabel('候选模态数');
yyaxis right;
plot(f_ghz, diag_mode_spread * 1e9, '.', 'Color', [0.90 0.50 0.12], 'MarkerSize', 4);
ylabel('模态时延极差 (ns)');
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
title('(f) 候选模态竞争程度');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

% --- (d) 多配置一致性 ---
subplot(3, 2, 6);
plot(consist_f_probe / 1e9, consist_tau_iqr * 1e9, '.', ...
    'Color', [0.18 0.62 0.38], 'MarkerSize', 5);
hold on;
xline(f_flat_lo, '--', 'Color', [0.5 0.5 0.5]);
xline(f_flat_hi, '--', 'Color', [0.5 0.5 0.5]);
xline(f_right_lo, ':', 'Color', [0.8 0.2 0.2]);
hold off;
ylabel('\tau IQR (ns)');
title('(d) 多窗口配置一致性（IQR 越大越不一致）');
set(gca, 'FontName', 'SimHei', 'FontSize', 9);
xlim([36.45 37.55]); grid on;

sgtitle('右侧边缘提取困难成因诊断', 'FontSize', 14, 'FontName', 'SimHei');

%% 8. 分区统计汇总
fprintf('\n===== 分区定量统计 =====\n');
regions = {
    'left_edge',     36.50e9, 36.62e9;
    'left_shoulder', 36.62e9, 36.78e9;
    'flat_mid',      36.78e9, 37.22e9;
    'right_shoulder',37.22e9, 37.38e9;
    'right_edge',    37.38e9, 37.55e9;
};

fprintf('  %-16s  %6s  %8s  %8s  %6s  %8s  %9s\n', ...
    'region', 'N_pts', 'RMS(mV)', 'eigGap', 'nMode', 'spread', 'boundary');
fprintf('  %s\n', repmat('-', 1, 76));

for ir = 1:size(regions, 1)
    mask = diag_f_probe >= regions{ir, 2} & diag_f_probe < regions{ir, 3};
    if ~any(mask), continue; end

    fprintf('  %-16s  %6d  %8.3f  %8.1f  %6.1f  %8.2f  %9.3f\n', ...
        regions{ir, 1}, sum(mask), ...
        median(diag_rms(mask)) * 1e3, ...
        median(diag_eig_gap(mask), 'omitnan'), ...
        median(diag_n_modes(mask)), ...
        median(diag_mode_spread(mask)) * 1e9, ...
        median(diag_boundary(mask)));
end

% 多配置一致性分区统计
fprintf('\n  %-16s  %6s  %8s  %8s\n', 'region', 'N_pts', 'IQR(ns)', 'STD(ns)');
fprintf('  %s\n', repmat('-', 1, 44));
for ir = 1:size(regions, 1)
    mask = consist_f_probe >= regions{ir, 2} & consist_f_probe < regions{ir, 3};
    if ~any(mask), continue; end
    fprintf('  %-16s  %6d  %8.3f  %8.3f\n', ...
        regions{ir, 1}, sum(mask), ...
        median(consist_tau_iqr(mask)) * 1e9, ...
        median(consist_tau_std(mask)) * 1e9);
end

fprintf('\n===== 诊断完成 =====\n');
