%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时域数据时延轨迹提取（自适应窗口 MDL-ESPRIT）
% 用途：
% 1. 先用固定窗口获得粗时延轨迹
% 2. 再根据粗轨迹在通带中段使用更长窗口、边缘使用更短窗口
% 3. 对比固定窗口与自适应窗口的散点覆盖情况
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist('data_file', 'var')
    incoming_data_file = data_file;
else
    incoming_data_file = '';
end
if exist('out_suffix', 'var')
    incoming_out_suffix = out_suffix;
else
    incoming_out_suffix = '';
end
if exist('case_title', 'var')
    incoming_case_title = case_title;
else
    incoming_case_title = '';
end
if exist('batch_summaries', 'var')
    incoming_batch_summaries = batch_summaries;
else
    incoming_batch_summaries = [];
end

clc;
clearvars -except incoming_data_file incoming_out_suffix incoming_case_title ...
    incoming_batch_summaries i_case data_file_list case_title_list out_suffix_list ...
    batch_summaries result_summary;
close all;

data_file = incoming_data_file;
out_suffix = incoming_out_suffix;
case_title = incoming_case_title;
batch_summaries = incoming_batch_summaries;

if isempty(data_file)
    data_file = 'lowpassfilter_filter.csv';
end
if isempty(out_suffix)
    out_suffix = '';
end
if isempty(case_title)
    [~, case_title, ~] = fileparts(data_file);
end

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
if ~contains(data_file, filesep)
    data_file = fullfile(script_dir, 'data', data_file);
end

%% 1. 实验 LFMCW 参数
f_start = 34e9;
f_end   = 37e9;
B       = f_end - f_start;
T_m     = 50e-6;
K       = B / T_m;

fprintf('===== 实测时延轨迹提取（自适应窗口） =====\n');
fprintf('LFMCW: %.0f-%.0f GHz, B=%.1f GHz, T_m=%.0f us, K=%.2e Hz/s\n', ...
    f_start/1e9, f_end/1e9, B/1e9, T_m*1e6, K);
fprintf('数据标签: %s\n', case_title);

%% 2. 数据文件与算法参数
cfg_base.win_len  = 150;
cfg_base.step_len = 13;
cfg_base.L_sub    = 75;
cfg_base.name     = '固定窗口';

cfg_adapt.step_center = 13;
cfg_adapt.win_short   = 120;
cfg_adapt.win_mid1    = 160;
cfg_adapt.win_mid2    = 220;
cfg_adapt.win_long    = 300;
cfg_adapt.tau_thr_1   = 0.9e-9;
cfg_adapt.tau_thr_2   = 1.4e-9;
cfg_adapt.tau_thr_3   = 2.0e-9;
cfg_adapt.name        = '自适应窗口';

cfg_hybrid.f_flat_lo = 36.8e9;
cfg_hybrid.f_flat_hi = 37.2e9;
cfg_hybrid.name      = '混合结果';

cfg_refine.enable         = true;
cfg_refine.band_lo        = 37.10e9;
cfg_refine.band_hi        = 37.50e9;
cfg_refine.win_len        = 150;
cfg_refine.step_len       = 6;
cfg_refine.min_freq_gap   = 0.012e9;
cfg_refine.tau_tol        = 0.35e-9;
cfg_refine.name           = '右侧边缘加密';

cfg_debug.enable          = true;
cfg_debug.left_lo         = 36.50e9;
cfg_debug.left_hi         = 36.80e9;
cfg_debug.right_lo        = 37.20e9;
cfg_debug.right_hi        = 37.50e9;
cfg_debug.max_print       = 24;
cfg_debug.csv_name        = ['exp_delay_trajectory_debug_records', out_suffix, '.csv'];

cfg_plot.tau_ref_max      = 1.97e-9;
cfg_plot.edge_color       = [0.16, 0.46, 0.72];
cfg_plot.mid_color        = [0.90, 0.50, 0.12];
cfg_plot.warn_color       = [0.72, 0.18, 0.18];
cfg_plot.ref_color        = [0.32, 0.32, 0.32];

f_hp_cut   = 10e3;
f_valid_lo = 20e3;
f_beat_max = 300e3;

fprintf('\n【步骤1】加载数据并进行周期平均...\n');
data = readmatrix(data_file);
t_raw = data(:, 1);
v_raw = data(:, 2);

dt = median(diff(t_raw));
fs = round(1 / dt);
N_total = length(t_raw);

fprintf('  数据文件: %s\n', data_file);
fprintf('  采样率: %.0f MHz, 总点数: %d, 时长: %.2f ms\n', ...
    fs/1e6, N_total, (t_raw(end) - t_raw(1)) * 1e3);

v_raw = v_raw - mean(v_raw);
N_per   = round(T_m * fs);
N_sweep = floor(N_total / N_per);
v_mat   = reshape(v_raw(1:N_sweep * N_per), N_per, N_sweep);
v_avg   = mean(v_mat, 2);

fprintf('  每周期 %d 点, %d 个完整周期, 叠加平均 SNR +%.1f dB\n', ...
    N_per, N_sweep, 10*log10(N_sweep));

%% 3. 降采样与高通预处理
fprintf('\n【步骤2】降采样与高通预处理...\n');
ds = 10;
v_ds    = v_avg(1:ds:end);
fs_proc = fs / ds;

[b_hp, a_hp] = butter(2, f_hp_cut / (fs_proc / 2), 'high');
v_proc = filtfilt(b_hp, a_hp, v_ds);
t_proc = (0:length(v_proc)-1).' / fs_proc;
rms_thr = max(abs(v_proc)) * 0.01;

fprintf('  降采样 x%d -> fs_proc=%.0f MHz, N_proc=%d\n', ...
    ds, fs_proc/1e6, length(v_proc));
fprintf('  高通滤波: >%.0f kHz, RMS 门限 = %.3e\n', ...
    f_hp_cut/1e3, rms_thr);

%% 4. 固定窗口粗提取
fprintf('\n【步骤3】固定窗口粗提取...\n');
base_raw = run_single_scale_extraction( ...
    v_proc, t_proc, fs_proc, f_start, K, rms_thr, ...
    cfg_base.win_len, cfg_base.step_len, cfg_base.L_sub, ...
    f_valid_lo, f_beat_max, true, cfg_base.name);

base_clean = postprocess_points(base_raw, true, cfg_base.name);
base_cal   = calibrate_frequency_axis(base_clean, K, true, cfg_base.name);

%% 5. 自适应窗口提取
fprintf('\n【步骤4】根据粗轨迹分配自适应窗口...\n');
adapt_raw = run_adaptive_extraction( ...
    v_proc, t_proc, fs_proc, f_start, K, rms_thr, ...
    base_clean, cfg_adapt, f_valid_lo, f_beat_max);

adapt_clean = postprocess_points(adapt_raw, true, cfg_adapt.name);
adapt_cal   = calibrate_frequency_axis(adapt_clean, K, true, cfg_adapt.name);

%% 5b. 混合策略：边缘保留固定窗口，中段采用自适应窗口
fprintf('\n【步骤4b】生成混合轨迹：边缘固定窗口 + 中段自适应窗口...\n');
hybrid_cal = fuse_hybrid_result(base_cal, adapt_clean, cfg_hybrid, true);

if cfg_refine.enable
    fprintf('\n【步骤4c】右侧边缘固定窗口加密...\n');
    hybrid_cal = refine_edge_with_dense_fixed( ...
        hybrid_cal, v_proc, t_proc, fs_proc, ...
        f_start, K, rms_thr, cfg_base, base_cal, ...
        cfg_refine, f_valid_lo, f_beat_max, true);
end

if cfg_debug.enable
    fprintf('\n【步骤4d】打印调试记录并导出 CSV...\n');
    print_debug_records(base_cal, hybrid_cal, cfg_debug);
end

%% 6. 汇总输出
fprintf('\n===== 固定窗口 vs 自适应窗口 =====\n');
fprintf('  固定窗口: 原始 %d 点 -> 后处理 %d 点\n', ...
    numel(base_raw.f_probe), numel(base_cal.f_probe));
fprintf('  自适应窗口: 原始 %d 点 -> 后处理 %d 点\n', ...
    numel(adapt_raw.f_probe), numel(adapt_cal.f_probe));
fprintf('  自适应新增净增散点: %+d\n', ...
    numel(adapt_cal.f_probe) - numel(base_cal.f_probe));

bins_cmp = [36.5, 36.7, 36.9, 37.1, 37.3, 37.5];
hist_base  = histcounts(base_cal.f_probe / 1e9, bins_cmp);
hist_adapt = histcounts(adapt_cal.f_probe / 1e9, bins_cmp);

fprintf('  对比频段: [36.5,36.7] [36.7,36.9] [36.9,37.1] [37.1,37.3] [37.3,37.5] GHz\n');
fprintf('  固定窗口: %4d %4d %4d %4d %4d\n', hist_base);
fprintf('  自适应窗: %4d %4d %4d %4d %4d\n', hist_adapt);

hist_hybrid = histcounts(hybrid_cal.f_probe / 1e9, bins_cmp);
fprintf('  混合结果: %4d %4d %4d %4d %4d\n', hist_hybrid);
fprintf('  混合策略频段: 中段 [%.2f, %.2f] GHz 采用自适应窗口，其余频段保留固定窗口\n', ...
    cfg_hybrid.f_flat_lo / 1e9, cfg_hybrid.f_flat_hi / 1e9);

%% 7. 绘图
fprintf('\n【步骤5】绘制对比图...\n');
figure('Color', 'w', 'Position', [100, 100, 980, 560]);
hold on;

scatter(base_cal.f_probe / 1e9, base_cal.tau * 1e9, 32, ...
    [0.70 0.70 0.70], 'filled', ...
    'MarkerFaceAlpha', 0.35, 'MarkerEdgeColor', 'none');

s_adapt = scatter(adapt_cal.f_probe / 1e9, adapt_cal.tau * 1e9, 46, ...
    adapt_cal.win_len, 'filled', ...
    'MarkerFaceAlpha', 0.85, 'MarkerEdgeColor', [0.15 0.15 0.15], ...
    'LineWidth', 0.4);

[f_sort, sort_idx] = sort(adapt_cal.f_probe);
tau_sort = adapt_cal.tau(sort_idx);

hold off;
grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('实测 LFMCW 时延轨迹：固定窗口与自适应窗口对比 [%s]', case_title), ...
    'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);
colormap(turbo);
cb = colorbar;
cb.Label.String = '自适应窗口长度 (sample)';
cb.FontSize = 10;
xlim([36.0, 38.0]);
legend([s_adapt], {'自适应窗口散点'}, 'Location', 'northeast', 'FontSize', 11);

export_thesis_figure(gcf, ['exp_delay_trajectory_adaptive', out_suffix], 14, 300);

figure('Color', 'w', 'Position', [120, 120, 940, 540]);
hold on;
yl_patch = [min(hybrid_cal.tau)*1e9 - 0.10, max(hybrid_cal.tau)*1e9 + 0.10];
h_patch = patch( ...
    [cfg_hybrid.f_flat_lo, cfg_hybrid.f_flat_hi, cfg_hybrid.f_flat_hi, cfg_hybrid.f_flat_lo] / 1e9, ...
    [yl_patch(1), yl_patch(1), yl_patch(2), yl_patch(2)], ...
    [0.96 0.95 0.88], ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.55, ...
    'HandleVisibility', 'off');

h_base = scatter(base_cal.f_probe / 1e9, base_cal.tau * 1e9, 30, ...
    [0.78 0.78 0.78], 'filled', ...
    'MarkerFaceAlpha', 0.18, 'MarkerEdgeColor', 'none');

mask_hybrid_mid = hybrid_cal.f_probe >= cfg_hybrid.f_flat_lo & ...
                  hybrid_cal.f_probe <= cfg_hybrid.f_flat_hi;
mask_hybrid_edge = ~mask_hybrid_mid;
mask_physical = hybrid_cal.tau <= cfg_plot.tau_ref_max;
mask_suspect  = ~mask_physical;

h_edge = scatter(hybrid_cal.f_probe(mask_hybrid_edge & mask_physical) / 1e9, ...
    hybrid_cal.tau(mask_hybrid_edge & mask_physical) * 1e9, 48, ...
    cfg_plot.edge_color, 'filled', ...
    'MarkerFaceAlpha', 0.92, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
    'LineWidth', 0.4);

h_mid = scatter(hybrid_cal.f_probe(mask_hybrid_mid & mask_physical) / 1e9, ...
    hybrid_cal.tau(mask_hybrid_mid & mask_physical) * 1e9, 54, ...
    cfg_plot.mid_color, 'filled', ...
    'MarkerFaceAlpha', 0.96, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
    'LineWidth', 0.4);

h_suspect = scatter(hybrid_cal.f_probe(mask_suspect) / 1e9, ...
    hybrid_cal.tau(mask_suspect) * 1e9, 76, ...
    'o', 'MarkerEdgeColor', cfg_plot.warn_color, ...
    'MarkerFaceColor', 'none', 'LineWidth', 1.4);
h_ref = yline(cfg_plot.tau_ref_max * 1e9, '--', 'Color', cfg_plot.ref_color, ...
    'LineWidth', 1.1, 'Label', '\tau = 1.97 ns', ...
    'LabelHorizontalAlignment', 'left', ...
    'LabelVerticalAlignment', 'bottom');
xline(cfg_hybrid.f_flat_lo / 1e9, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.0);
xline(cfg_hybrid.f_flat_hi / 1e9, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.0);
text(36.58, yl_patch(2) - 0.06, '边缘区固定窗口', ...
    'FontName', 'SimHei', 'FontSize', 10, 'Color', [0.20 0.20 0.20]);
text((cfg_hybrid.f_flat_lo + cfg_hybrid.f_flat_hi) / 2 / 1e9, yl_patch(2) - 0.06, ...
    '平坦区自适应补点', ...
    'FontName', 'SimHei', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'Color', [0.20 0.20 0.20]);
hold off;

grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('实测 LFMCW 时延轨迹（边缘固定窗口 + 平坦区自适应补点） [%s]', case_title), ...
    'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);
xlim([36.45, 37.55]);
ylim([max(0, yl_patch(1)), max(2.05, yl_patch(2))]);
legend([h_base, h_edge, h_mid, h_suspect, h_ref], ...
    {'固定窗口参考散点', '边缘区最终散点', '平坦区最终散点', ...
     '超过 1.97 ns 的点', '1.97 ns 参考线'}, ...
    'Location', 'northeast', 'FontSize', 11);

export_thesis_figure(gcf, ['exp_delay_trajectory_hybrid', out_suffix], 14, 300);
export_thesis_figure(gcf, ['exp_delay_trajectory_merged', out_suffix], 14, 300);

fprintf('\n频率轴已根据滤波器通带 36.5-37.5 GHz 完成工程校准。\n');

result_summary = struct();
result_summary.case_title = case_title;
result_summary.data_file = data_file;
result_summary.base_points = numel(base_cal.f_probe);
result_summary.adapt_points = numel(adapt_cal.f_probe);
result_summary.hybrid_points = numel(hybrid_cal.f_probe);
result_summary.left_edge_points = sum(hybrid_cal.f_probe >= cfg_debug.left_lo & hybrid_cal.f_probe <= cfg_debug.left_hi);
result_summary.right_edge_points = sum(hybrid_cal.f_probe >= cfg_debug.right_lo & hybrid_cal.f_probe <= cfg_debug.right_hi);
result_summary.base_anchor_lo_ghz = base_cal.f_anchor_lo / 1e9;
result_summary.base_anchor_hi_ghz = base_cal.f_anchor_hi / 1e9;
result_summary.base_a_cal = base_cal.a_cal;
result_summary.hybrid_tau_min_ns = min(hybrid_cal.tau) * 1e9;
result_summary.hybrid_tau_max_ns = max(hybrid_cal.tau) * 1e9;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 局部函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = run_single_scale_extraction( ...
    v_proc, t_proc, fs_proc, f_start, K, rms_thr, ...
    win_len, step_len, L_sub, f_valid_lo, f_beat_max, ...
    show_summary, tag_name)

N_proc  = length(v_proc);
num_win = floor((N_proc - win_len) / step_len) + 1;

f_probe_arr  = zeros(num_win, 1);
tau_arr      = zeros(num_win, 1);
amp_arr      = zeros(num_win, 1);
center_arr   = zeros(num_win, 1);
win_len_arr  = zeros(num_win, 1);
cnt = 0;

n_skip_rms   = 0;
n_skip_valid = 0;
n_skip_bound = 0;

for i = 1:num_win
    idx = (i-1) * step_len + 1 : (i-1) * step_len + win_len;
    if idx(end) > N_proc
        break;
    end

    [is_ok, f_probe, tau_val, amp_val, center_idx, reason_code] = ...
        process_single_window(v_proc(idx), idx, t_proc, fs_proc, ...
        f_start, K, rms_thr, L_sub, f_valid_lo, f_beat_max);

    if ~is_ok
        switch reason_code
            case 1
                n_skip_rms = n_skip_rms + 1;
            case 2
                n_skip_valid = n_skip_valid + 1;
            case 3
                n_skip_bound = n_skip_bound + 1;
        end
        continue;
    end

    cnt = cnt + 1;
    f_probe_arr(cnt) = f_probe;
    tau_arr(cnt)     = tau_val;
    amp_arr(cnt)     = amp_val;
    center_arr(cnt)  = center_idx;
    win_len_arr(cnt) = win_len;
end

out.f_probe    = f_probe_arr(1:cnt);
out.tau        = tau_arr(1:cnt);
out.amp        = amp_arr(1:cnt);
out.center_idx = center_arr(1:cnt);
out.win_len    = win_len_arr(1:cnt);

if show_summary
    fprintf('  %s: win=%d (%.1f us), step=%d, L_sub=%d, 共 %d 窗口\n', ...
        tag_name, win_len, win_len / fs_proc * 1e6, step_len, L_sub, num_win);
    fprintf('  原始散点: %d / %d, 跳过: RMS=%d, 无有效频率=%d, 超物理上界=%d\n', ...
        cnt, num_win, n_skip_rms, n_skip_valid, n_skip_bound);
end
end

function out = run_adaptive_extraction( ...
    v_proc, t_proc, fs_proc, f_start, K, rms_thr, ...
    base_clean, cfg_adapt, f_valid_lo, f_beat_max)

N_proc = length(v_proc);
center_grid = (round(cfg_adapt.win_short/2)+1 : cfg_adapt.step_center : ...
    N_proc - round(cfg_adapt.win_short/2)).';

if isempty(base_clean.f_probe)
    error('固定窗口粗轨迹为空，无法进行自适应窗口分配。');
end

[f_base_sort, sort_idx] = sort(base_clean.f_probe);
tau_base_sort = base_clean.tau(sort_idx);
span_smooth = max(5, 2 * floor(numel(tau_base_sort) / 12) + 1);
if mod(span_smooth, 2) == 0
    span_smooth = span_smooth + 1;
end
tau_base_smooth = movmean(tau_base_sort, span_smooth);

f_grid = f_start + K * t_proc(center_grid);
tau_pred = interp1(f_base_sort, tau_base_smooth, f_grid, 'linear', 'extrap');

N_grid = numel(center_grid);
f_probe_arr  = zeros(N_grid, 1);
tau_arr      = zeros(N_grid, 1);
amp_arr      = zeros(N_grid, 1);
center_arr   = zeros(N_grid, 1);
win_len_arr  = zeros(N_grid, 1);
tau_pred_arr = zeros(N_grid, 1);
cnt = 0;

n_short = 0;
n_mid1  = 0;
n_mid2  = 0;
n_long  = 0;

for i = 1:N_grid
    center_idx = center_grid(i);
    tau_now = tau_pred(i);

    if tau_now < cfg_adapt.tau_thr_1
        win_len = cfg_adapt.win_long;
        n_long = n_long + 1;
    elseif tau_now < cfg_adapt.tau_thr_2
        win_len = cfg_adapt.win_mid2;
        n_mid2 = n_mid2 + 1;
    elseif tau_now < cfg_adapt.tau_thr_3
        win_len = cfg_adapt.win_mid1;
        n_mid1 = n_mid1 + 1;
    else
        win_len = cfg_adapt.win_short;
        n_short = n_short + 1;
    end

    idx_start = center_idx - floor(win_len / 2);
    idx_start = max(1, min(idx_start, N_proc - win_len + 1));
    idx = idx_start : idx_start + win_len - 1;
    L_sub = round(win_len / 2);

    [is_ok, f_probe, tau_val, amp_val, center_new, reason_code] = ...
        process_single_window(v_proc(idx), idx, t_proc, fs_proc, ...
        f_start, K, rms_thr, L_sub, f_valid_lo, f_beat_max);

    if ~is_ok
        if reason_code == 2 && win_len < cfg_adapt.win_long
            win_try = min(cfg_adapt.win_long, N_proc);
            idx_start = center_idx - floor(win_try / 2);
            idx_start = max(1, min(idx_start, N_proc - win_try + 1));
            idx = idx_start : idx_start + win_try - 1;
            L_sub = round(win_try / 2);

            [is_ok, f_probe, tau_val, amp_val, center_new, ~] = ...
                process_single_window(v_proc(idx), idx, t_proc, fs_proc, ...
                f_start, K, rms_thr, L_sub, f_valid_lo, f_beat_max);
            if is_ok
                win_len = win_try;
            end
        end
    end

    if ~is_ok
        continue;
    end

    cnt = cnt + 1;
    f_probe_arr(cnt)  = f_probe;
    tau_arr(cnt)      = tau_val;
    amp_arr(cnt)      = amp_val;
    center_arr(cnt)   = center_new;
    win_len_arr(cnt)  = win_len;
    tau_pred_arr(cnt) = tau_now;
end

f_probe_arr  = f_probe_arr(1:cnt);
tau_arr      = tau_arr(1:cnt);
amp_arr      = amp_arr(1:cnt);
center_arr   = center_arr(1:cnt);
win_len_arr  = win_len_arr(1:cnt);
tau_pred_arr = tau_pred_arr(1:cnt);

[center_unique, ia] = unique(center_arr, 'stable');

out.f_probe    = f_probe_arr(ia);
out.tau        = tau_arr(ia);
out.amp        = amp_arr(ia);
out.center_idx = center_unique;
out.win_len    = win_len_arr(ia);
out.tau_pred   = tau_pred_arr(ia);

fprintf('  自适应中心网格: %d 个\n', N_grid);
fprintf('  预测时延阈值: <%.2f ns -> %d, <%.2f ns -> %d, <%.2f ns -> %d, 其余 -> %d\n', ...
    cfg_adapt.tau_thr_1*1e9, cfg_adapt.win_long, ...
    cfg_adapt.tau_thr_2*1e9, cfg_adapt.win_mid2, ...
    cfg_adapt.tau_thr_3*1e9, cfg_adapt.win_mid1, ...
    cfg_adapt.win_short);
fprintf('  窗口分配统计: short=%d, mid1=%d, mid2=%d, long=%d\n', ...
    n_short, n_mid1, n_mid2, n_long);
fprintf('  自适应原始散点: %d / %d\n', numel(out.f_probe), N_grid);
end

function [is_ok, f_probe, tau_val, amp_val, center_idx, reason_code] = ...
    process_single_window(x_win, idx, t_proc, fs_proc, f_start, K, ...
    rms_thr, L_sub, f_valid_lo, f_beat_max)

is_ok = false;
f_probe = NaN;
tau_val = NaN;
amp_val = rms(x_win);
center_idx = idx(round(numel(idx)/2));
reason_code = 0;

if amp_val < rms_thr
    reason_code = 1;
    return;
end

win_len = numel(x_win);
M_sub = win_len - L_sub + 1;
if M_sub <= 2
    reason_code = 2;
    return;
end

X_h = zeros(L_sub, M_sub);
for k = 1:M_sub
    X_h(:, k) = x_win(k : k+L_sub-1).';
end

R_fwd = (X_h * X_h') / M_sub;
J = fliplr(eye(L_sub));
R_x = (R_fwd + J * conj(R_fwd) * J) / 2;

[V, D] = eig(R_x);
[lam, id] = sort(diag(D), 'descend');
V = V(:, id);

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

Us = V(:, 1:num_s);
Phi = (Us(1:end-1,:)' * Us(1:end-1,:)) \ ...
      (Us(1:end-1,:)' * Us(2:end,:));
eig_vals = eig(Phi);
est_f = abs(angle(eig_vals)) * fs_proc / (2*pi);

valid_mask = est_f > f_valid_lo & est_f < fs_proc / 4;
est_f = est_f(valid_mask);
if isempty(est_f)
    reason_code = 2;
    return;
end

proj_power = zeros(numel(est_f), 1);
for jj = 1:numel(est_f)
    steering = exp(1j * 2*pi * est_f(jj) / fs_proc * (0:L_sub-1).');
    proj_power(jj) = abs(steering' * V(:, 1))^2;
end
[~, rank_idx] = sort(proj_power, 'descend');

f_beat = [];
for jj = rank_idx.'
    if est_f(jj) <= f_beat_max
        f_beat = est_f(jj);
        break;
    end
end

if isempty(f_beat)
    reason_code = 3;
    return;
end

t_c = t_proc(center_idx);
f_probe = f_start + K * t_c;
tau_val = f_beat / K;
is_ok = true;
end

function out = postprocess_points(in, show_summary, tag_name)
if isempty(in.f_probe)
    error('%s 提取结果为空。', tag_name);
end

amp_norm = in.amp / (max(in.amp) + eps);
mask_amp = amp_norm > 0.15;

tau_masked = in.tau(mask_amp);
tau_q25 = prctile(tau_masked, 25);
tau_q75 = prctile(tau_masked, 75);
tau_iqr = tau_q75 - tau_q25;
tau_lo  = tau_q25 - 2.0 * tau_iqr;
tau_hi  = tau_q75 + 2.0 * tau_iqr;
mask_iqr = in.tau >= tau_lo & in.tau <= tau_hi;

[f_sorted, sort_idx] = sort(in.f_probe);
tau_sorted = in.tau(sort_idx);
local_span = max(5, 2 * floor(numel(in.tau) / 40) + 1);
if mod(local_span, 2) == 0
    local_span = local_span + 1;
end
tau_med = movmedian(tau_sorted, local_span);
tau_dev = abs(tau_sorted - tau_med);
dev_thr = max(3 * 1.4826 * movmedian(tau_dev, local_span), 0.3e-9);
mask_local = true(numel(in.tau), 1);
mask_local(sort_idx) = tau_dev <= dev_thr;

mask_clean = mask_amp & mask_iqr & mask_local;

out.f_probe    = in.f_probe(mask_clean);
out.tau        = in.tau(mask_clean);
out.amp        = in.amp(mask_clean);
out.center_idx = in.center_idx(mask_clean);
out.win_len    = in.win_len(mask_clean);

if isfield(in, 'tau_pred')
    out.tau_pred = in.tau_pred(mask_clean);
end

if show_summary
    fprintf('  %s 后处理: 原始 %d -> 幅度 %d -> +IQR %d -> +连续性 %d\n', ...
        tag_name, numel(in.tau), sum(mask_amp), ...
        sum(mask_amp & mask_iqr), sum(mask_clean));
end
end

function out = calibrate_frequency_axis(in, K, show_summary, tag_name)
if numel(in.f_probe) < 6
    error('%s 清洗后散点过少，无法进行双锚点校准。', tag_name);
end

f_edge_lo = 36.5e9;
f_edge_hi = 37.5e9;

[f_sorted, sort_idx] = sort(in.f_probe);
tau_sorted = in.tau(sort_idx);
win_sorted = in.win_len(sort_idx);
amp_sorted = in.amp(sort_idx);

N_half = round(numel(f_sorted) / 2);
tau_left_smooth  = movmean(tau_sorted(1:N_half), ...
    max(3, round(N_half / 5)));
tau_right_smooth = movmean(tau_sorted(N_half+1:end), ...
    max(3, round((numel(f_sorted) - N_half) / 5)));

[~, idx_lo] = max(tau_left_smooth);
[~, idx_hi_rel] = max(tau_right_smooth);
idx_hi = N_half + idx_hi_rel;

f_anchor_lo = f_sorted(idx_lo);
f_anchor_hi = f_sorted(idx_hi);

a_cal = (f_edge_hi - f_edge_lo) / (f_anchor_hi - f_anchor_lo);
b_cal = f_edge_lo - a_cal * f_anchor_lo;

out.f_probe = a_cal * f_sorted + b_cal;
out.tau     = tau_sorted;
out.amp     = amp_sorted;
out.win_len = win_sorted;
if isfield(in, 'source_code')
    out.source_code = in.source_code(sort_idx);
end
out.a_cal   = a_cal;
out.b_cal   = b_cal;
out.f_anchor_lo = f_anchor_lo;
out.f_anchor_hi = f_anchor_hi;

if show_summary
    fprintf('  %s 校准: 左锚点 %.3f GHz -> 36.5 GHz, 右锚点 %.3f GHz -> 37.5 GHz\n', ...
        tag_name, f_anchor_lo / 1e9, f_anchor_hi / 1e9);
    fprintf('  %s 校准系数: a=%.4f, b=%.3f GHz, 有效K=%.2e Hz/s\n', ...
        tag_name, a_cal, b_cal / 1e9, K * a_cal);
    fprintf('  %s 最终范围: f = %.2f-%.2f GHz, tau = %.2f-%.2f ns\n', ...
        tag_name, min(out.f_probe)/1e9, max(out.f_probe)/1e9, ...
        min(out.tau)*1e9, max(out.tau)*1e9);
end
end

function out = fuse_hybrid_result(base_cal, adapt_clean, cfg_hybrid, show_summary)
f_adapt_on_base = base_cal.a_cal * adapt_clean.f_probe + base_cal.b_cal;

mask_base_edge = base_cal.f_probe < cfg_hybrid.f_flat_lo | ...
                 base_cal.f_probe > cfg_hybrid.f_flat_hi;
mask_adapt_mid = f_adapt_on_base >= cfg_hybrid.f_flat_lo & ...
                 f_adapt_on_base <= cfg_hybrid.f_flat_hi;

f_merge   = [base_cal.f_probe(mask_base_edge); f_adapt_on_base(mask_adapt_mid)];
tau_merge = [base_cal.tau(mask_base_edge); adapt_clean.tau(mask_adapt_mid)];
amp_merge = [base_cal.amp(mask_base_edge); adapt_clean.amp(mask_adapt_mid)];
win_merge = [base_cal.win_len(mask_base_edge); adapt_clean.win_len(mask_adapt_mid)];
src_merge = [ones(sum(mask_base_edge), 1); 2 * ones(sum(mask_adapt_mid), 1)];

[f_merge, sort_idx] = sort(f_merge);
tau_merge = tau_merge(sort_idx);
amp_merge = amp_merge(sort_idx);
win_merge = win_merge(sort_idx);
src_merge = src_merge(sort_idx);

out.f_probe = f_merge;
out.tau     = tau_merge;
out.amp     = amp_merge;
out.win_len = win_merge;
out.source_code = src_merge;

if show_summary
    fprintf('  混合频段: [%.2f, %.2f] GHz 使用自适应窗口\n', ...
        cfg_hybrid.f_flat_lo / 1e9, cfg_hybrid.f_flat_hi / 1e9);
    fprintf('  固定窗口保留边缘点: %d\n', sum(mask_base_edge));
    fprintf('  自适应窗口中段点: %d\n', sum(mask_adapt_mid));
    fprintf('  混合结果总点数: %d\n', numel(out.f_probe));
    fprintf('  混合结果范围: f = %.2f-%.2f GHz, tau = %.2f-%.2f ns\n', ...
        min(out.f_probe)/1e9, max(out.f_probe)/1e9, ...
        min(out.tau)*1e9, max(out.tau)*1e9);
end
end

function out = refine_edge_with_dense_fixed( ...
    hybrid_cal, v_proc, t_proc, fs_proc, ...
    f_start, K, rms_thr, cfg_base, base_cal, ...
    cfg_refine, f_valid_lo, f_beat_max, show_summary)

dense_raw = run_single_scale_extraction( ...
    v_proc, t_proc, fs_proc, f_start, K, rms_thr, ...
    cfg_refine.win_len, cfg_refine.step_len, cfg_base.L_sub, ...
    f_valid_lo, f_beat_max, false, cfg_refine.name);

dense_clean = postprocess_points(dense_raw, false, cfg_refine.name);
f_dense_cal = base_cal.a_cal * dense_clean.f_probe + base_cal.b_cal;

mask_band = f_dense_cal >= cfg_refine.band_lo & f_dense_cal <= cfg_refine.band_hi;
f_add   = f_dense_cal(mask_band);
tau_add = dense_clean.tau(mask_band);
amp_add = dense_clean.amp(mask_band);
win_add = dense_clean.win_len(mask_band);

mask_ref = base_cal.f_probe >= cfg_refine.band_lo & base_cal.f_probe <= cfg_refine.band_hi;
if sum(mask_ref) < 3
    mask_ref = base_cal.f_probe >= (cfg_refine.band_lo - 0.10e9) & ...
               base_cal.f_probe <= cfg_refine.band_hi;
end

f_ref = base_cal.f_probe(mask_ref);
tau_ref = base_cal.tau(mask_ref);
[f_ref, sort_ref] = sort(f_ref);
tau_ref = tau_ref(sort_ref);

tau_pred = interp1(f_ref, tau_ref, f_add, 'linear', 'extrap');
mask_consistent = abs(tau_add - tau_pred) <= cfg_refine.tau_tol;

f_add   = f_add(mask_consistent);
tau_add = tau_add(mask_consistent);
amp_add = amp_add(mask_consistent);
win_add = win_add(mask_consistent);

is_new = false(size(f_add));
for i = 1:numel(f_add)
    if isempty(hybrid_cal.f_probe) || ...
       min(abs(hybrid_cal.f_probe - f_add(i))) > cfg_refine.min_freq_gap
        is_new(i) = true;
    end
end

f_merge   = [hybrid_cal.f_probe; f_add(is_new)];
tau_merge = [hybrid_cal.tau; tau_add(is_new)];
amp_merge = [hybrid_cal.amp; amp_add(is_new)];
win_merge = [hybrid_cal.win_len; win_add(is_new)];
src_merge = [hybrid_cal.source_code; 3 * ones(sum(is_new), 1)];

[f_merge, sort_idx] = sort(f_merge);
tau_merge = tau_merge(sort_idx);
amp_merge = amp_merge(sort_idx);
win_merge = win_merge(sort_idx);
src_merge = src_merge(sort_idx);

out.f_probe = f_merge;
out.tau     = tau_merge;
out.amp     = amp_merge;
out.win_len = win_merge;
out.source_code = src_merge;

if show_summary
    fprintf('  加密频段: [%.2f, %.2f] GHz\n', ...
        cfg_refine.band_lo / 1e9, cfg_refine.band_hi / 1e9);
    fprintf('  加密窗口: win=%d, step=%d\n', ...
        cfg_refine.win_len, cfg_refine.step_len);
    fprintf('  固定窗口参考点: %d, 一致性阈值: %.2f ns\n', ...
        numel(f_ref), cfg_refine.tau_tol * 1e9);
    fprintf('  候选加密点: %d, 一致性保留: %d, 实际新增: %d, 去重阈值: %.0f MHz\n', ...
        sum(mask_band), numel(f_add), sum(is_new), cfg_refine.min_freq_gap / 1e6);
    fprintf('  加密后总点数: %d\n', numel(out.f_probe));
end
end

function print_debug_records(base_cal, hybrid_cal, cfg_debug)
mask_left  = hybrid_cal.f_probe >= cfg_debug.left_lo  & hybrid_cal.f_probe <= cfg_debug.left_hi;
mask_right = hybrid_cal.f_probe >= cfg_debug.right_lo & hybrid_cal.f_probe <= cfg_debug.right_hi;

fprintf('===== 调试记录：左右边缘对比 =====\n');
print_region_summary('左侧边缘', hybrid_cal, mask_left);
print_region_summary('右侧边缘', hybrid_cal, mask_right);

tau_left_med = median_or_nan(hybrid_cal.tau(mask_left)) * 1e9;
tau_right_med = median_or_nan(hybrid_cal.tau(mask_right)) * 1e9;
fprintf('  左右边缘中位数差: Δtau = %.3f ns (右-左)\n', tau_right_med - tau_left_med);
fprintf('  固定窗口校准锚点: 左 %.3f GHz, 右 %.3f GHz, a=%.4f, 有效K=%.2e Hz/s\n', ...
    base_cal.f_anchor_lo / 1e9, base_cal.f_anchor_hi / 1e9, ...
    base_cal.a_cal, base_cal.a_cal * 6.00e13);

fprintf('\n--- 左侧边缘明细（最终结果）---\n');
print_point_table(hybrid_cal, mask_left, cfg_debug.max_print);
fprintf('\n--- 右侧边缘明细（最终结果）---\n');
print_point_table(hybrid_cal, mask_right, cfg_debug.max_print);

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
out_dir = fullfile(project_root, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
csv_path = fullfile(out_dir, cfg_debug.csv_name);
export_debug_csv(csv_path, hybrid_cal, mask_left, mask_right);
fprintf('【导出】%s\n', csv_path);
end

function print_region_summary(region_name, data_struct, region_mask)
tau_ns = data_struct.tau(region_mask) * 1e9;
f_ghz = data_struct.f_probe(region_mask) / 1e9;
src = data_struct.source_code(region_mask);

fprintf('  %s: 点数=%d, f=[%.3f, %.3f] GHz, tau中位=%.3f ns, tau均值=%.3f ns\n', ...
    region_name, sum(region_mask), ...
    min_or_nan(f_ghz), max_or_nan(f_ghz), ...
    median_or_nan(tau_ns), mean_or_nan(tau_ns));
fprintf('    来源分布: 固定边缘=%d, 中段自适应=%d, 右侧加密=%d\n', ...
    sum(src == 1), sum(src == 2), sum(src == 3));
end

function print_point_table(data_struct, region_mask, max_print)
idx_all = find(region_mask);
if isempty(idx_all)
    fprintf('  （无记录）\n');
    return;
end

[~, order] = sort(data_struct.f_probe(idx_all), 'ascend');
idx_all = idx_all(order);
N_show = min(numel(idx_all), max_print);

fprintf('  %-4s %-8s %-12s %-10s %-8s\n', '序号', '来源', 'f(GHz)', 'tau(ns)', 'win');
for i = 1:N_show
    ii = idx_all(i);
    fprintf('  %-4d %-8s %-12.3f %-10.3f %-8d\n', ...
        i, source_code_to_label(data_struct.source_code(ii)), ...
        data_struct.f_probe(ii) / 1e9, ...
        data_struct.tau(ii) * 1e9, ...
        round(data_struct.win_len(ii)));
end

if numel(idx_all) > N_show
    fprintf('  ... 其余 %d 条已写入 CSV\n', numel(idx_all) - N_show);
end
end

function export_debug_csv(csv_path, data_struct, mask_left, mask_right)
fid = fopen(csv_path, 'w');
if fid < 0
    warning('无法写入调试 CSV: %s', csv_path);
    return;
end

cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, 'index,region,source,f_GHz,tau_ns,win_len\n');

for i = 1:numel(data_struct.f_probe)
    if mask_left(i)
        region_name = 'left_edge';
    elseif mask_right(i)
        region_name = 'right_edge';
    else
        region_name = 'other';
    end

    fprintf(fid, '%d,%s,%s,%.6f,%.6f,%d\n', ...
        i, region_name, source_code_to_label(data_struct.source_code(i)), ...
        data_struct.f_probe(i) / 1e9, ...
        data_struct.tau(i) * 1e9, ...
        round(data_struct.win_len(i)));
end
end

function out = source_code_to_label(source_code)
switch source_code
    case 1
        out = 'fixed';
    case 2
        out = 'adapt';
    case 3
        out = 'refine';
    otherwise
        out = 'unknown';
end
end

function out = median_or_nan(x)
if isempty(x)
    out = NaN;
else
    out = median(x);
end
end

function out = mean_or_nan(x)
if isempty(x)
    out = NaN;
else
    out = mean(x);
end
end

function out = min_or_nan(x)
if isempty(x)
    out = NaN;
else
    out = min(x);
end
end

function out = max_or_nan(x)
if isempty(x)
    out = NaN;
else
    out = max(x);
end
end

function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 300; end

height_cm = width_cm * 0.618;
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
out_dir = fullfile(project_root, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

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

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
fprintf('【导出】%s\n', file_tiff);
end
