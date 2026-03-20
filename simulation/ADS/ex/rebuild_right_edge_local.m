function out = rebuild_right_edge_local(hybrid_cal, v_proc, t_proc, fs_proc, ...
    f_start, K, rms_thr, base_cal, cfg, f_valid_lo, f_beat_max, show_summary)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Right-edge rebuild driven only by right-side local reference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

esp = esprit_extract();

% 1. Build a local right-side reference from trusted points near band_lo.
ref_lo = cfg.band_lo - 0.12e9;
ref_hi = cfg.band_lo;
mask_ref = hybrid_cal.f_probe >= ref_lo & hybrid_cal.f_probe <= ref_hi;

if sum(mask_ref) < 3
    mask_ref = base_cal.f_probe >= ref_lo & base_cal.f_probe <= ref_hi;
end

[f_ref, si] = sort(hybrid_cal.f_probe(mask_ref));
tau_ref = hybrid_cal.tau(mask_ref);
tau_ref = tau_ref(si);

if numel(f_ref) < 2
    out = hybrid_cal;
    if show_summary
        fprintf('  [右侧局部参考] 点数不足，跳过重建\n');
    end
    return;
end

[f_ref, ui] = unique(f_ref, 'stable');
tau_ref = tau_ref(ui);

if numel(tau_ref) >= 3
    sp = max(3, 2*floor(numel(tau_ref)/4)+1);
    if mod(sp,2)==0, sp = sp+1; end
    tau_ref = movmedian(tau_ref, sp);
end

poly_deg = min(2, numel(f_ref) - 1);
p_ref = polyfit(f_ref, tau_ref, poly_deg);

if show_summary
    fprintf('  [右侧局部参考] f = %.3f-%.3f GHz, tau = %.2f-%.2f ns, %d 个点\n', ...
        min(f_ref)/1e9, max(f_ref)/1e9, min(tau_ref)*1e9, max(tau_ref)*1e9, numel(f_ref));
end

% 2. Purge existing right-edge points inconsistent with the local reference.
mask_in = hybrid_cal.f_probe >= cfg.purge_band_lo & hybrid_cal.f_probe <= cfg.band_hi;
mask_keep = true(numel(hybrid_cal.f_probe), 1);

if any(mask_in)
    tau_pred_old = local_predict(hybrid_cal.f_probe(mask_in), f_ref, tau_ref, p_ref);
    bad = abs(hybrid_cal.tau(mask_in) - tau_pred_old) > cfg.purge_tol;
    idx_in = find(mask_in);
    mask_keep(idx_in(bad)) = false;
end

n_purged = sum(~mask_keep);
f_kept = hybrid_cal.f_probe(mask_keep);
tau_kept = hybrid_cal.tau(mask_keep);
amp_kept = hybrid_cal.amp(mask_keep);
win_kept = hybrid_cal.win_len(mask_keep);
src_kept = hybrid_cal.source_code(mask_keep);

if show_summary
    fprintf('  [清洗旧点] 频段内 %d 个, 剔除 %d 个\n', sum(mask_in), n_purged);
end

% 3. Multi-window / multi-L_sub candidate extraction on the right edge.
f_all = [];
tau_all = [];
amp_all = [];
win_all = [];

N_proc = length(v_proc);
L_sub_ratios = cfg.L_sub_ratios;

for iw = 1:numel(cfg.win_lens)
    wl = cfg.win_lens(iw);
    centers = (round(wl/2)+1 : cfg.step_len : N_proc-round(wl/2));

    for ir = 1:numel(L_sub_ratios)
        Ls = max(4, round(wl * L_sub_ratios(ir)));
        nc = 0;

        for ic = 1:numel(centers)
            ci = centers(ic);
            is = max(1, min(ci-floor(wl/2), N_proc-wl+1));
            idx = is : is+wl-1;

            [fp, ta, av, ~, ~] = esp.all_modes( ...
                v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
                rms_thr, Ls, f_valid_lo, f_beat_max);

            if isempty(ta), continue; end

            fc = base_cal.a_cal * fp + base_cal.b_cal;
            if fc < cfg.band_lo || fc > cfg.band_hi, continue; end

            te = local_predict(fc, f_ref, tau_ref, p_ref);
            [~, bi] = min(abs(ta - te));

            f_all = [f_all; fc];
            tau_all = [tau_all; ta(bi)];
            amp_all = [amp_all; av];
            win_all = [win_all; wl];
            nc = nc + 1;
        end

        if show_summary
            fprintf('  win=%d, L_sub=%d (%.0f%%): %d 个\n', ...
                wl, Ls, L_sub_ratios(ir)*100, nc);
        end
    end
end

% 4. Merge near-frequency duplicates, keep the candidate closest to reference.
if numel(f_all) > 1
    tau_pred_all = local_predict(f_all, f_ref, tau_ref, p_ref);
    dev_all = abs(tau_all - tau_pred_all);

    [f_all, si] = sort(f_all);
    tau_all = tau_all(si);
    amp_all = amp_all(si);
    win_all = win_all(si);
    dev_all = dev_all(si);

    keep = true(numel(f_all), 1);
    ii = 1;
    while ii <= numel(f_all)
        jj = ii;
        while jj < numel(f_all) && (f_all(jj+1) - f_all(ii)) < 0.004e9
            jj = jj + 1;
        end
        if jj > ii
            g = ii:jj;
            [~, best] = min(dev_all(g));
            mk = false(numel(g), 1);
            mk(best) = true;
            keep(g(~mk)) = false;
        end
        ii = jj + 1;
    end

    f_all = f_all(keep);
    tau_all = tau_all(keep);
    amp_all = amp_all(keep);
    win_all = win_all(keep);
end

% 5. Consistency screening against the local right-side reference.
n_candidates = numel(f_all);
if ~isempty(f_all)
    tau_pred = local_predict(f_all, f_ref, tau_ref, p_ref);
    mc = (tau_all >= tau_pred - cfg.tau_tol_lo) & ...
         (tau_all <= tau_pred + cfg.tau_tol_hi);
    f_all = f_all(mc);
    tau_all = tau_all(mc);
    amp_all = amp_all(mc);
    win_all = win_all(mc);
end

% 6. Append only genuinely new points.
is_new = false(size(f_all));
for i = 1:numel(f_all)
    if isempty(f_kept) || min(abs(f_kept - f_all(i))) > cfg.min_freq_gap
        is_new(i) = true;
    end
end

f_m = [f_kept; f_all(is_new)];
tau_m = [tau_kept; tau_all(is_new)];
amp_m = [amp_kept; amp_all(is_new)];
win_m = [win_kept; win_all(is_new)];
src_m = [src_kept; 3*ones(sum(is_new),1)];

[f_m, si] = sort(f_m);
tau_m = tau_m(si);
amp_m = amp_m(si);
win_m = win_m(si);
src_m = src_m(si);

out.f_probe = f_m;
out.tau = tau_m;
out.amp = amp_m;
out.win_len = win_m;
out.source_code = src_m;

if show_summary
    fprintf('  重建: 剔除 %d, 候选 %d, 一致性保留 %d, 新增 %d, 总计 %d\n', ...
        n_purged, n_candidates, numel(f_all), sum(is_new), numel(out.f_probe));
end
end

function tau_pred = local_predict(f_query, f_ref, tau_ref, p_ref)
tau_pred = interp1(f_ref, tau_ref, f_query, 'linear', NaN);
mask_out = isnan(tau_pred);
if any(mask_out)
    tau_pred(mask_out) = polyval(p_ref, f_query(mask_out));
end
end
