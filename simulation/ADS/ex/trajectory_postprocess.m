function fn = trajectory_postprocess()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 轨迹后处理函数集（清洗、频率轴校准、混合融合）
% 用法：fn = trajectory_postprocess(); 然后 fn.clean(...), fn.calibrate(...) 等
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fn.clean     = @postprocess_points;
fn.calibrate = @calibrate_frequency_axis;
fn.fuse      = @fuse_hybrid_result;
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
end
end
