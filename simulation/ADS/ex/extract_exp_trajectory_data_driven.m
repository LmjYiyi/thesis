%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时延轨迹提取（纯数据驱动版）
% 右侧边缘重建完全基于局部连续性与多窗口共识，不使用任何镜像先验。
% 依赖文件：esprit_extract.m, trajectory_postprocess.m, rebuild_right_edge_local.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));

esp = esprit_extract();
pp  = trajectory_postprocess();

%% 1. LFMCW 参数
f_start = 34e9;
f_end   = 37e9;
B       = f_end - f_start;
T_m     = 50e-6;
K       = B / T_m;

fprintf('===== 实测时延轨迹提取（纯数据驱动版） =====\n');
fprintf('LFMCW: %.0f-%.0f GHz, B=%.1f GHz, T_m=%.0f us, K=%.2e Hz/s\n', ...
    f_start/1e9, f_end/1e9, B/1e9, T_m*1e6, K);

%% 2. 加载数据
data_file = fullfile(script_dir, 'data', 'lowpassfilter_filter.csv');
fprintf('\n正在加载数据...\n');
data = readmatrix(data_file);
t_raw = data(:, 1);
v_raw = data(:, 2);

dt = median(diff(t_raw));
fs = round(1 / dt);
N_total = length(t_raw);

fprintf('  采样率: %.0f MHz, 总点数: %d, 时长: %.2f ms\n', ...
    fs/1e6, N_total, (t_raw(end) - t_raw(1)) * 1e3);

%% 3. 周期叠加平均
v_raw = v_raw - mean(v_raw);
N_per   = round(T_m * fs);
N_sweep = floor(N_total / N_per);
v_mat   = reshape(v_raw(1:N_sweep * N_per), N_per, N_sweep);
v_avg   = mean(v_mat, 2);

fprintf('  每周期 %d 点, %d 个完整周期, 叠加平均 SNR+%.1f dB\n', ...
    N_per, N_sweep, 10*log10(N_sweep));

%% 4. 降采样与高通预处理
ds = 10;
v_ds    = v_avg(1:ds:end);
fs_proc = fs / ds;

f_hp_cut = 10e3;
[b_hp, a_hp] = butter(2, f_hp_cut / (fs_proc / 2), 'high');
v_proc = filtfilt(b_hp, a_hp, v_ds);
f_refine_lp = 0.80e6;
[b_refine, a_refine] = butter(4, f_refine_lp / (fs_proc / 2), 'low');
v_proc_refine = filtfilt(b_refine, a_refine, v_proc);
t_proc = (0:length(v_proc)-1).' / fs_proc;
rms_thr = max(abs(v_proc)) * 0.01;

fprintf('  降采样 x%d -> fs_proc=%.0f MHz, N_proc=%d\n', ...
    ds, fs_proc/1e6, length(v_proc));

%% 5. 配置
cfg_base.win_len  = 150;
cfg_base.step_len = 13;
cfg_base.L_sub    = 75;
cfg_base.name     = '固定窗口';

cfg_adapt.step_center = 9;
cfg_adapt.win_short   = 120;
cfg_adapt.win_mid1    = 160;
cfg_adapt.win_mid2    = 220;
cfg_adapt.win_long    = 300;
cfg_adapt.tau_thr_1   = 0.9e-9;
cfg_adapt.tau_thr_2   = 1.4e-9;
cfg_adapt.tau_thr_3   = 2.0e-9;
cfg_adapt.name        = '自适应窗口';

cfg_hybrid.f_flat_lo    = 36.78e9;
cfg_hybrid.f_flat_hi    = 37.38e9;
cfg_hybrid.mid_fill_gap = 0.020e9;

cfg_refine.enable        = true;
cfg_refine.band_lo       = 37.38e9;
cfg_refine.band_hi       = 37.50e9;
cfg_refine.win_lens      = [130, 100, 80];
cfg_refine.L_sub_ratios  = [1/2, 1/3, 2/5];
cfg_refine.step_len      = 3;
cfg_refine.min_freq_gap  = 0.003e9;
cfg_refine.group_freq_gap = 0.004e9;
cfg_refine.ref_span_lo    = 0.24e9;
cfg_refine.ref_span_hi    = 0.00e9;
cfg_refine.ref_min_points = 5;
cfg_refine.consensus_min  = 2;
cfg_refine.edge_uplift_gain = 0.95;
cfg_refine.edge_uplift_power = 0.85;
cfg_refine.edge_uplift_cap = 0.28e-9;
cfg_refine.tau_tol_lo    = 0.32e-9;
cfg_refine.tau_tol_hi    = 0.36e-9;
cfg_refine.purge_band_lo = 37.30e9;
cfg_refine.purge_tol     = 0.36e-9;
cfg_refine.name          = '右侧局部连续性重建';

f_valid_lo = 20e3;
f_beat_max = 300e3;

%% 6. 固定窗口粗提取
fprintf('\n【步骤1】固定窗口粗提取...\n');
base_raw = esp.run_fixed(v_proc, t_proc, fs_proc, f_start, K, ...
    rms_thr, cfg_base, f_valid_lo, f_beat_max, true);

base_clean = pp.clean(base_raw, true, cfg_base.name);
base_cal   = pp.calibrate(base_clean, K, true, cfg_base.name);

%% 7. 自适应分窗
fprintf('\n【步骤2】自适应分窗...\n');
adapt_raw = esp.run_adaptive(v_proc, t_proc, fs_proc, f_start, K, ...
    rms_thr, base_clean, cfg_adapt, f_valid_lo, f_beat_max);

adapt_clean = pp.clean(adapt_raw, true, cfg_adapt.name);

%% 8. 混合融合
fprintf('\n【步骤3】混合融合...\n');
hybrid_cal = pp.fuse(base_cal, adapt_clean, cfg_hybrid, true);

%% 9. 右侧局部连续性重建
if cfg_refine.enable
    fprintf('\n【步骤4】右侧局部连续性重建...\n');
    hybrid_cal = rebuild_right_edge_local(hybrid_cal, v_proc_refine, t_proc, fs_proc, ...
        f_start, K, rms_thr, base_cal, cfg_refine, f_valid_lo, f_beat_max, true);
end

%% 10. 汇总输出
fprintf('\n===== 结果汇总（纯数据驱动版） =====\n');
fprintf('  固定窗口: %d 点, 自适应: %d 点, 最终: %d 点\n', ...
    numel(base_cal.f_probe), numel(adapt_clean.f_probe), numel(hybrid_cal.f_probe));
fprintf('  频率: %.3f - %.3f GHz\n', ...
    min(hybrid_cal.f_probe)/1e9, max(hybrid_cal.f_probe)/1e9);
fprintf('  时延: %.2f - %.2f ns, 中位数 %.2f ns\n', ...
    min(hybrid_cal.tau)*1e9, max(hybrid_cal.tau)*1e9, median(hybrid_cal.tau)*1e9);

mask_right = hybrid_cal.f_probe > cfg_hybrid.f_flat_hi;
if any(mask_right)
    fprintf('\n  -- 右侧虚线右边 (>%.2f GHz) --\n', cfg_hybrid.f_flat_hi/1e9);
    fprintf('  点数: %d\n', sum(mask_right));
    fprintf('  频率: %.3f - %.3f GHz\n', ...
        min(hybrid_cal.f_probe(mask_right))/1e9, ...
        max(hybrid_cal.f_probe(mask_right))/1e9);
    fprintf('  时延: %.2f - %.2f ns, 中位数 %.2f ns\n', ...
        min(hybrid_cal.tau(mask_right))*1e9, ...
        max(hybrid_cal.tau(mask_right))*1e9, ...
        median(hybrid_cal.tau(mask_right))*1e9);
end

mask_seg = hybrid_cal.f_probe >= 37.30e9 & hybrid_cal.f_probe <= 37.50e9;
if any(mask_seg)
    fprintf('  37.30-37.50 GHz 段: %d 点, 中位数 %.2f ns, 最大 %.2f ns\n', ...
        sum(mask_seg), median(hybrid_cal.tau(mask_seg))*1e9, max(hybrid_cal.tau(mask_seg))*1e9);
end
print_delay_shape_diagnostics(hybrid_cal, 'data_driven');

%% 11. 绘图
figure('Color', 'w', 'Position', [100, 100, 960, 540]);
hold on;

likely_curve = likely_filter_delay_curve(hybrid_cal, script_dir);
fprintf('\n[likely filter curve] passband %.2f-%.2f GHz, tau_mid=%.2f ns, tau_peak=%.2f ns\n', ...
    min(likely_curve.f_ghz), max(likely_curve.f_ghz), ...
    likely_curve.tau_floor_ns, likely_curve.tau_peak_ns);

scatter(base_cal.f_probe / 1e9, base_cal.tau * 1e9, 28, ...
    [0.80 0.80 0.80], 'filled', ...
    'MarkerFaceAlpha', 0.22, 'MarkerEdgeColor', 'none');

mask_base  = hybrid_cal.source_code == 1;
mask_adapt = hybrid_cal.source_code == 2;
mask_dense = hybrid_cal.source_code == 3;

h_base = scatter(hybrid_cal.f_probe(mask_base) / 1e9, ...
    hybrid_cal.tau(mask_base) * 1e9, 42, ...
    [0.16 0.46 0.72], 'filled', ...
    'MarkerFaceAlpha', 0.90, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
    'LineWidth', 0.4);
h_adapt = scatter(hybrid_cal.f_probe(mask_adapt) / 1e9, ...
    hybrid_cal.tau(mask_adapt) * 1e9, 50, ...
    [0.90 0.50 0.12], 'filled', ...
    'MarkerFaceAlpha', 0.94, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
    'LineWidth', 0.4);
h_dense = scatter(hybrid_cal.f_probe(mask_dense) / 1e9, ...
    hybrid_cal.tau(mask_dense) * 1e9, 58, ...
    [0.18 0.62 0.38], 'filled', ...
    'MarkerFaceAlpha', 0.96, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
    'LineWidth', 0.4);
h_likely = plot(likely_curve.f_ghz, likely_curve.tau_ns, '-', ...
    'Color', [0.78 0.12 0.12], 'LineWidth', 2.2);
set(get(get(h_likely, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');

xline(cfg_hybrid.f_flat_lo / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
xline(cfg_hybrid.f_flat_hi / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
hold off;

grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title('实测 LFMCW 时延轨迹（纯数据驱动版）', 'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);
xlim([36.45, 37.55]);
legend([h_base, h_adapt, h_dense], ...
    {'边缘区固定窗口', '中段自适应窗口', '右侧局部连续性重建'}, ...
    'Location', 'northeast', 'FontSize', 11);

export_thesis_figure(gcf, 'exp_delay_trajectory_data_driven', 14, 300);
fprintf('\n频率轴已根据纯数据驱动锚点完成工程校准。\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_delay_shape_diagnostics(in, tag_name)
fprintf('\n===== Shape Diagnostics (%s) =====\n', tag_name);

[f_s, si] = sort(in.f_probe);
tau_s = in.tau(si);

fprintf('  idx   f_probe(GHz)   tau(ns)   region           source\n');
for i = 1:numel(f_s)
    tau_ns = tau_s(i) * 1e9;
    region_name = classify_region(f_s(i));
    source_name = decode_source(in, si(i));
    fprintf('  %3d   %10.3f   %7.3f   %-14s %-8s\n', ...
        i, f_s(i)/1e9, tau_ns, region_name, source_name);
end
end

function region_name = classify_region(f_val)
if f_val < 36.62e9
    region_name = 'left_edge';
elseif f_val < 36.78e9
    region_name = 'left_shoulder';
elseif f_val <= 37.22e9
    region_name = 'flat_mid';
elseif f_val < 37.38e9
    region_name = 'right_shoulder';
else
    region_name = 'right_edge';
end
end

function source_name = decode_source(in, idx)
source_name = 'n/a';
if ~isfield(in, 'source_code')
    return;
end

switch in.source_code(idx)
    case 1
        source_name = 'base';
    case 2
        source_name = 'adapt';
    case 3
        source_name = 'rebuild';
    otherwise
        source_name = 'unknown';
end
end

function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 3, width_cm = 14; end
if nargin < 4, dpi = 300; end

height_cm = width_cm * 0.618;
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures_export');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

set(fig_handle, 'Color', 'w', 'Units', 'centimeters', ...
    'Position', [2 2 width_cm height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPosition', [0 0 width_cm height_cm], ...
    'PaperSize', [width_cm height_cm]);

for ax = findall(fig_handle, 'Type', 'axes').'
    set(ax, 'FontName', 'SimHei', 'FontSize', 10, 'LineWidth', 1.0, ...
        'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.20, 'TickDir', 'out');
end
for ln = findall(fig_handle, 'Type', 'line').'
    if strcmp(get(ln, 'LineStyle'), 'none')
        set(ln, 'LineWidth', 1.0);
    else
        set(ln, 'LineWidth', 1.5);
    end
end

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
fprintf('【导出】%s\n', file_tiff);
end
