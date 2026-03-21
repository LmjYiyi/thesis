function fn = trajectory_postprocess()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 轨迹后处理函数集（清洗、频率轴校准、混合融合）
% 用法：fn = trajectory_postprocess(); 然后 fn.clean(...), fn.calibrate(...) 等
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fn.clean     = @postprocess_points;
fn.calibrate = @calibrate_frequency_axis;
fn.fuse      = @fuse_hybrid_result;
fn.define_regions    = @define_regions;
fn.classify_region   = @classify_region;
fn.decode_source     = @decode_source;
fn.export_figure     = @export_thesis_figure;
fn.print_diagnostics = @print_delay_shape_diagnostics;
end

%% ---- 后处理：幅度/IQR/连续性清洗 ----
function out = postprocess_points(in, show_summary, tag_name)
if isempty(in.f_probe), error('%s 提取结果为空。', tag_name); end

amp_norm = in.amp / (max(in.amp) + eps);
mask_amp = amp_norm > 0.15;

tau_m = in.tau(mask_amp);
tau_iqr = prctile(tau_m, 75) - prctile(tau_m, 25);
mask_iqr = in.tau >= (prctile(tau_m,25) - 2*tau_iqr) & ...
           in.tau <= (prctile(tau_m,75) + 2*tau_iqr);

[f_s, si] = sort(in.f_probe);
tau_s = in.tau(si);
sp = max(5, 2*floor(numel(in.tau)/40)+1);
if mod(sp,2)==0, sp = sp+1; end
tau_med = movmedian(tau_s, sp);
tau_dev = abs(tau_s - tau_med);
dev_thr = max(3 * 1.4826 * movmedian(tau_dev, sp), 0.3e-9);
mask_local = true(numel(in.tau), 1);
mask_local(si) = tau_dev <= dev_thr;

mask = mask_amp & mask_iqr & mask_local;
out.f_probe = in.f_probe(mask); out.tau = in.tau(mask);
out.amp = in.amp(mask); out.center_idx = in.center_idx(mask);
out.win_len = in.win_len(mask);
if isfield(in, 'tau_pred'), out.tau_pred = in.tau_pred(mask); end

if show_summary
    fprintf('  %s 后处理: 原始 %d -> 幅度 %d -> +IQR %d -> +连续性 %d\n', ...
        tag_name, numel(in.tau), sum(mask_amp), sum(mask_amp & mask_iqr), sum(mask));
end
end

%% ---- 频率轴校准（左锚点 + 对称匹配右锚点） ----
function out = calibrate_frequency_axis(in, K, show_summary, tag_name)
if numel(in.f_probe) < 6
    error('%s 清洗后散点过少，无法校准。', tag_name);
end

f_edge_lo = 36.5e9;  BW = 1.0e9;

[f_s, si] = sort(in.f_probe);
tau_s = in.tau(si); win_s = in.win_len(si); amp_s = in.amp(si);

N_half = round(numel(f_s) / 2);

% 左锚点：左半 tau 峰值
tau_left_sm = movmean(tau_s(1:N_half), max(3, round(N_half/5)));
[~, idx_lo] = max(tau_left_sm);
f_anchor_lo = f_s(idx_lo);
tau_anchor_lo = tau_s(idx_lo);

% 右锚点：右半中从外向内找第一个 tau >= 0.90 * 左锚点 tau 的位置
right_tau = tau_s(N_half+1:end);
right_sm = movmean(right_tau, max(3, round(numel(right_tau)/5)));
idx_hi_rel = numel(right_tau);
for rr = numel(right_tau):-1:1
    if right_sm(rr) >= tau_anchor_lo * 0.90
        idx_hi_rel = rr; break;
    end
end
f_anchor_hi = f_s(N_half + idx_hi_rel);

a_cal = BW / (f_anchor_hi - f_anchor_lo);
b_cal = f_edge_lo - a_cal * f_anchor_lo;

out.f_probe = a_cal * f_s + b_cal;
out.tau = tau_s; out.amp = amp_s; out.win_len = win_s;
if isfield(in, 'source_code'), out.source_code = in.source_code(si); end
out.a_cal = a_cal; out.b_cal = b_cal;
out.f_anchor_lo = f_anchor_lo; out.f_anchor_hi = f_anchor_hi;

if show_summary
    fprintf('  %s 校准: 左锚点 %.3f GHz (tau=%.2f ns) -> 36.5 GHz\n', ...
        tag_name, f_anchor_lo/1e9, tau_anchor_lo*1e9);
    fprintf('  %s 校准: 右锚点(对称匹配) %.3f GHz -> 37.5 GHz\n', ...
        tag_name, f_anchor_hi/1e9);
    fprintf('  %s 校准系数: a=%.4f, b=%.3f GHz, 有效K=%.2e Hz/s\n', ...
        tag_name, a_cal, b_cal/1e9, K*a_cal);
    fprintf('  %s 最终范围: f = %.2f-%.2f GHz, tau = %.2f-%.2f ns\n', ...
        tag_name, min(out.f_probe)/1e9, max(out.f_probe)/1e9, ...
        min(out.tau)*1e9, max(out.tau)*1e9);
end
end

%% ---- 混合融合：边缘用固定窗，中段用自适应 ----
function out = fuse_hybrid_result(base_cal, adapt_clean, cfg, show_summary)
f_adapt_cal = base_cal.a_cal * adapt_clean.f_probe + base_cal.b_cal;

mask_edge = base_cal.f_probe < cfg.f_flat_lo | base_cal.f_probe > cfg.f_flat_hi;
mask_mid  = ~mask_edge;
mask_adapt = f_adapt_cal >= cfg.f_flat_lo & f_adapt_cal <= cfg.f_flat_hi;

f_bm = base_cal.f_probe(mask_mid); tau_bm = base_cal.tau(mask_mid);
amp_bm = base_cal.amp(mask_mid);   win_bm = base_cal.win_len(mask_mid);
f_am = f_adapt_cal(mask_adapt);     tau_am = adapt_clean.tau(mask_adapt);
amp_am = adapt_clean.amp(mask_adapt); win_am = adapt_clean.win_len(mask_adapt);
[tau_am, clamp_report] = clamp_local_adapt_overshoot(f_am, tau_am, base_cal, cfg);

% 中段空洞用固定窗补填
if isempty(f_am)
    fill = true(size(f_bm));
else
    fill = false(size(f_bm));
    for i = 1:numel(f_bm)
        if min(abs(f_am - f_bm(i))) > cfg.mid_fill_gap, fill(i) = true; end
    end
end

f_merge = [base_cal.f_probe(mask_edge); f_am; f_bm(fill)];
tau_merge = [base_cal.tau(mask_edge); tau_am; tau_bm(fill)];
amp_merge = [base_cal.amp(mask_edge); amp_am; amp_bm(fill)];
win_merge = [base_cal.win_len(mask_edge); win_am; win_bm(fill)];
src_merge = [ones(sum(mask_edge),1); 2*ones(numel(f_am),1); ones(sum(fill),1)];

[f_merge, si] = sort(f_merge);
tau_merge = tau_merge(si); amp_merge = amp_merge(si);
win_merge = win_merge(si); src_merge = src_merge(si);

out.f_probe = f_merge; out.tau = tau_merge; out.amp = amp_merge;
out.win_len = win_merge; out.source_code = src_merge;

if show_summary
    fprintf('  混合频段: [%.2f, %.2f] GHz\n', cfg.f_flat_lo/1e9, cfg.f_flat_hi/1e9);
    fprintf('  固定窗口边缘: %d, 自适应中段: %d, 总点数: %d\n', ...
        sum(mask_edge), sum(mask_adapt), numel(out.f_probe));
    if clamp_report.n_clamped > 0
        fprintf('  adapt local cap: clamped %d pts, max drop %.3f ns\n', ...
            clamp_report.n_clamped, clamp_report.max_drop_s * 1e9);
    end
end
end

%% ---- 区域定义（由 cfg.regions 生成边界表） ----
%% ---- Local cap for adaptive overshoot in selected bands ----
function [tau_out, report] = clamp_local_adapt_overshoot(f_adapt_hz, tau_in_s, base_cal, cfg)
tau_out = tau_in_s;
report.n_clamped = 0;
report.max_drop_s = 0;

if isempty(f_adapt_hz) || isempty(tau_in_s)
    return;
end

if ~isfield(cfg, 'local_cap') || ~isstruct(cfg.local_cap)
    return;
end

cap_cfg = cfg.local_cap;
if ~isfield(cap_cfg, 'enable') || ~cap_cfg.enable
    return;
end

if ~isfield(cap_cfg, 'bands_hz') || size(cap_cfg.bands_hz, 2) ~= 2
    return;
end

margin_default_s = 0;
if isfield(cap_cfg, 'margin_s') && ~isempty(cap_cfg.margin_s)
    margin_cfg = cap_cfg.margin_s;
else
    margin_cfg = margin_default_s;
end

ref_pad_hz = 0.08e9;
if isfield(cap_cfg, 'ref_pad_hz') && isfinite(cap_cfg.ref_pad_hz)
    ref_pad_hz = max(cap_cfg.ref_pad_hz, 0);
end

for ib = 1:size(cap_cfg.bands_hz, 1)
    f_lo = cap_cfg.bands_hz(ib, 1);
    f_hi = cap_cfg.bands_hz(ib, 2);
    margin_s = local_select_margin(margin_cfg, ib, margin_default_s);
    mask_band = f_adapt_hz >= f_lo & f_adapt_hz <= f_hi;
    if ~any(mask_band)
        continue;
    end

    [tau_cap_s, has_ref] = build_local_cap_from_base( ...
        f_adapt_hz(mask_band), base_cal, f_lo, f_hi, ref_pad_hz, margin_s);
    if ~has_ref
        continue;
    end

    tau_band_s = tau_out(mask_band);
    mask_clip = isfinite(tau_cap_s) & (tau_band_s > tau_cap_s);
    if ~any(mask_clip)
        continue;
    end

    tau_old_s = tau_band_s(mask_clip);
    tau_band_s(mask_clip) = tau_cap_s(mask_clip);
    tau_out(mask_band) = tau_band_s;

    drop_s = tau_old_s - tau_cap_s(mask_clip);
    report.n_clamped = report.n_clamped + sum(mask_clip);
    report.max_drop_s = max(report.max_drop_s, max(drop_s));
end
end

function [tau_cap_s, has_ref] = build_local_cap_from_base(f_query_hz, base_cal, ...
    f_lo, f_hi, ref_pad_hz, margin_s)
tau_cap_s = NaN(size(f_query_hz));
has_ref = false;

if ~isfield(base_cal, 'f_probe') || ~isfield(base_cal, 'tau') ...
        || isempty(base_cal.f_probe) || isempty(base_cal.tau)
    return;
end

mask_ref = base_cal.f_probe >= (f_lo - ref_pad_hz) & ...
           base_cal.f_probe <= (f_hi + ref_pad_hz) & ...
           isfinite(base_cal.f_probe) & isfinite(base_cal.tau);
if sum(mask_ref) < 4
    return;
end

[f_ref_hz, si] = sort(base_cal.f_probe(mask_ref));
tau_ref_s = base_cal.tau(mask_ref);
tau_ref_s = tau_ref_s(si);
[f_ref_hz, ui] = unique(f_ref_hz, 'stable');
tau_ref_s = tau_ref_s(ui);

for iq = 1:numel(f_query_hz)
    dq = abs(f_ref_hz - f_query_hz(iq));
    local_mask = dq <= ref_pad_hz;
    if sum(local_mask) < 3
        [~, idx_near] = mink(dq, min(3, numel(dq)));
        local_mask = false(size(dq));
        local_mask(idx_near) = true;
    end

    tau_local_s = tau_ref_s(local_mask);
    tau_local_s = tau_local_s(isfinite(tau_local_s));
    if numel(tau_local_s) < 3
        continue;
    end

    tau_cap_s(iq) = percentile_linear(tau_local_s, 0.30) + margin_s;
end

has_ref = any(isfinite(tau_cap_s));
end

function margin_s = local_select_margin(margin_cfg, idx, margin_default_s)
margin_s = margin_default_s;
if isempty(margin_cfg)
    return;
end

if isscalar(margin_cfg)
    if isfinite(margin_cfg)
        margin_s = max(margin_cfg, 0);
    end
    return;
end

if idx <= numel(margin_cfg) && isfinite(margin_cfg(idx))
    margin_s = max(margin_cfg(idx), 0);
end
end

function value = percentile_linear(x, p)
x = sort(x(:));
if isempty(x)
    value = NaN;
    return;
end

p = min(max(p, 0), 1);
if numel(x) == 1
    value = x;
    return;
end

pos = 1 + (numel(x) - 1) * p;
idx_lo = floor(pos);
idx_hi = ceil(pos);
alpha = pos - idx_lo;
value = (1 - alpha) * x(idx_lo) + alpha * x(idx_hi);
end

%% ---- 区域定义（由 cfg.regions 生成边界表） ----
function regions_table = define_regions(cfg)
r = cfg.regions;
regions_table = {
    'left_edge',      r.left_edge(1),      r.left_edge(2);
    'left_shoulder',  r.left_shoulder(1),   r.left_shoulder(2);
    'flat_mid',       r.flat_mid(1),        r.flat_mid(2);
    'right_shoulder', r.right_shoulder(1),  r.right_shoulder(2);
    'right_edge',     r.right_edge(1),      r.right_edge(2);
};
end

%% ---- 频率区域分类 ----
function region_name = classify_region(f_val, cfg)
r = cfg.regions;
if f_val < r.left_edge(2)
    region_name = 'left_edge';
elseif f_val < r.left_shoulder(2)
    region_name = 'left_shoulder';
elseif f_val <= r.flat_mid(2)
    region_name = 'flat_mid';
elseif f_val < r.right_shoulder(2)
    region_name = 'right_shoulder';
else
    region_name = 'right_edge';
end
end

%% ---- 来源代码解码 ----
function source_name = decode_source(in, idx)
source_name = 'n/a';
if ~isfield(in, 'source_code'), return; end
switch in.source_code(idx)
    case 1, source_name = 'base';
    case 2, source_name = 'adapt';
    case 3, source_name = 'rebuild';
    otherwise, source_name = 'unknown';
end
end

%% ---- 论文插图统一导出 ----
function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
if nargin < 3, width_cm = 14; end
if nargin < 4, dpi = 300; end

height_cm = width_cm * 0.618;
lib_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(fileparts(lib_dir), '..', 'figures_export');
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

%% ---- 时延形状诊断表 ----
function print_delay_shape_diagnostics(in, tag_name, cfg)
fprintf('\n===== Shape Diagnostics (%s) =====\n', tag_name);

[f_s, si] = sort(in.f_probe);
tau_s = in.tau(si);

fprintf('  idx   f_probe(GHz)   tau(ns)   region           source\n');
for i = 1:numel(f_s)
    tau_ns = tau_s(i) * 1e9;
    region_name = classify_region(f_s(i), cfg);
    source_name = decode_source(in, si(i));
    fprintf('  %3d   %10.3f   %7.3f   %-14s %-8s\n', ...
        i, f_s(i)/1e9, tau_ns, region_name, source_name);
end
end
