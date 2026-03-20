function out = rebuild_right_edge_local(hybrid_cal, v_proc, t_proc, fs_proc, ...
    f_start, K, rms_thr, base_cal, cfg, f_valid_lo, f_beat_max, show_summary)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Right-edge rebuild driven by local data only.
% Strategy:
% 1. Build a trusted local reference near the right plateau.
% 2. Reject old points inconsistent with that reference.
% 3. Extract multi-window candidates and keep only consensus-supported ones.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

esp = esprit_extract();

ref_span_lo = get_cfg(cfg, 'ref_span_lo', 0.14e9);
ref_span_hi = get_cfg(cfg, 'ref_span_hi', 0.02e9);
ref_min_points = get_cfg(cfg, 'ref_min_points', 6);
group_gap = get_cfg(cfg, 'group_freq_gap', 0.004e9);
consensus_min = get_cfg(cfg, 'consensus_min', 2);
edge_uplift_gain = get_cfg(cfg, 'edge_uplift_gain', 0.00);
edge_uplift_power = get_cfg(cfg, 'edge_uplift_power', 1.00);
edge_uplift_cap = get_cfg(cfg, 'edge_uplift_cap', 0.50e-9);

ref_hi = min(cfg.band_lo + ref_span_hi, cfg.purge_band_lo);
ref_lo = ref_hi - ref_span_lo;
[f_ref, tau_ref, ref_label] = build_reference(hybrid_cal, base_cal, ref_lo, ref_hi, ref_min_points);

if numel(f_ref) < 3
    out = hybrid_cal;
    if show_summary
        fprintf('  [right-local ref] insufficient support, skip rebuild\n');
    end
    return;
end

[f_ref, ui] = unique(f_ref, 'stable');
tau_ref = tau_ref(ui);

if numel(tau_ref) >= 5
    sp = max(5, 2 * floor(numel(tau_ref) / 5) + 1);
    if mod(sp, 2) == 0, sp = sp + 1; end
    tau_ref = movmedian(tau_ref, sp);
end

fit_ref = fit_local_reference(f_ref, tau_ref);
uplift_model = build_edge_uplift_model(f_ref, tau_ref, fit_ref, ...
    cfg.purge_band_lo, cfg.band_hi, ...
    edge_uplift_gain, edge_uplift_power, edge_uplift_cap);

if show_summary
    fprintf('  [right-local ref] %s, f = %.3f-%.3f GHz, tau = %.2f-%.2f ns, %d pts\n', ...
        ref_label, min(f_ref)/1e9, max(f_ref)/1e9, ...
        min(tau_ref)*1e9, max(tau_ref)*1e9, numel(f_ref));
    if uplift_model.enabled
        fprintf('  [edge uplift] local_hi=%.2f ns, target_hi=%.2f ns, delta=%.2f ns\n', ...
            uplift_model.tau_edge_local*1e9, uplift_model.tau_edge_target*1e9, ...
            uplift_model.delta_edge*1e9);
    end
end

mask_in = hybrid_cal.f_probe >= cfg.purge_band_lo & hybrid_cal.f_probe <= cfg.band_hi;
mask_keep = true(numel(hybrid_cal.f_probe), 1);

if any(mask_in)
    tau_pred_old = predict_target(hybrid_cal.f_probe(mask_in), f_ref, tau_ref, fit_ref, uplift_model);
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
    fprintf('  [purge old points] in-band %d, removed %d\n', sum(mask_in), n_purged);
end

f_all = [];
tau_all = [];
amp_all = [];
win_all = [];
cfg_id_all = [];

N_proc = length(v_proc);
L_sub_ratios = cfg.L_sub_ratios;

for iw = 1:numel(cfg.win_lens)
    wl = cfg.win_lens(iw);
    centers = (round(wl/2)+1 : cfg.step_len : N_proc-round(wl/2));

    for ir = 1:numel(L_sub_ratios)
        Ls = max(4, round(wl * L_sub_ratios(ir)));
        cfg_id = (iw - 1) * numel(L_sub_ratios) + ir;
        n_cfg = 0;

        for ic = 1:numel(centers)
            ci = centers(ic);
            is = max(1, min(ci - floor(wl/2), N_proc - wl + 1));
            idx = is : is + wl - 1;

            [fp, ta, av, ~, ~] = esp.all_modes( ...
                v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
                rms_thr, Ls, f_valid_lo, f_beat_max);

            if isempty(ta), continue; end

            fc = base_cal.a_cal * fp + base_cal.b_cal;
            if fc < cfg.band_lo || fc > cfg.band_hi, continue; end

            te = predict_target(fc, f_ref, tau_ref, fit_ref, uplift_model);
            [~, bi] = min(abs(ta - te));

            f_all(end+1, 1) = fc; %#ok<AGROW>
            tau_all(end+1, 1) = ta(bi); %#ok<AGROW>
            amp_all(end+1, 1) = av; %#ok<AGROW>
            win_all(end+1, 1) = wl; %#ok<AGROW>
            cfg_id_all(end+1, 1) = cfg_id; %#ok<AGROW>
            n_cfg = n_cfg + 1;
        end

        if show_summary
            fprintf('  win=%d, L_sub=%d (%.0f%%): %d pts\n', ...
                wl, Ls, L_sub_ratios(ir) * 100, n_cfg);
        end
    end
end

if numel(f_all) > 1
    tau_pred_all = predict_target(f_all, f_ref, tau_ref, fit_ref, uplift_model);
    dev_all = abs(tau_all - tau_pred_all);

    [f_all, si] = sort(f_all);
    tau_all = tau_all(si);
    amp_all = amp_all(si);
    win_all = win_all(si);
    cfg_id_all = cfg_id_all(si);
    dev_all = dev_all(si);

    keep = false(numel(f_all), 1);
    ii = 1;
    while ii <= numel(f_all)
        jj = ii;
        while jj < numel(f_all) && (f_all(jj+1) - f_all(ii)) < group_gap
            jj = jj + 1;
        end

        g = ii:jj;
        if numel(unique(cfg_id_all(g))) >= consensus_min
            [~, best] = min(dev_all(g));
            keep(g(best)) = true;
        end
        ii = jj + 1;
    end

    f_all = f_all(keep);
    tau_all = tau_all(keep);
    amp_all = amp_all(keep);
    win_all = win_all(keep);
end

n_candidates = numel(f_all);
if ~isempty(f_all)
    tau_pred = predict_target(f_all, f_ref, tau_ref, fit_ref, uplift_model);
    mc = (tau_all >= tau_pred - cfg.tau_tol_lo) & ...
         (tau_all <= tau_pred + cfg.tau_tol_hi);
    f_all = f_all(mc);
    tau_all = tau_all(mc);
    amp_all = amp_all(mc);
    win_all = win_all(mc);
end

if numel(f_all) >= 4
    sp = max(3, 2 * floor(numel(f_all) / 4) + 1);
    if mod(sp, 2) == 0, sp = sp + 1; end
    tau_med = movmedian(tau_all, sp);
    tau_base = max(tau_med, predict_target(f_all, f_ref, tau_ref, fit_ref, uplift_model));
    dev_lo = max(tau_base - tau_all, 0);
    dev_hi = max(tau_all - tau_base, 0);
    dev_thr_lo = max(2.5 * robust_mad_nonzero(dev_lo), 0.10e-9);
    dev_thr_hi = max(3.5 * robust_mad_nonzero(dev_hi), 0.18e-9);
    keep_local = (tau_all >= tau_base - dev_thr_lo) & ...
                 (tau_all <= tau_base + dev_thr_hi);
    f_all = f_all(keep_local);
    tau_all = tau_all(keep_local);
    amp_all = amp_all(keep_local);
    win_all = win_all(keep_local);
end

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
src_m = [src_kept; 3 * ones(sum(is_new), 1)];

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
    fprintf('  rebuild: purged %d, consensus %d, kept %d, added %d, total %d\n', ...
        n_purged, n_candidates, numel(f_all), sum(is_new), numel(out.f_probe));
end
end

function [f_ref, tau_ref, ref_label] = build_reference(hybrid_cal, base_cal, ref_lo, ref_hi, ref_min_points)
[f_ref, tau_ref] = extract_band_points(hybrid_cal, ref_lo, ref_hi);
ref_label = 'hybrid';

if numel(f_ref) < ref_min_points
    [f_ref, tau_ref] = extract_band_points(base_cal, ref_lo, ref_hi);
    ref_label = 'base';
end
end

function [f_ref, tau_ref] = extract_band_points(data_in, ref_lo, ref_hi)
mask_ref = data_in.f_probe >= ref_lo & data_in.f_probe <= ref_hi;
[f_ref, si] = sort(data_in.f_probe(mask_ref));
tau_ref = data_in.tau(mask_ref);
tau_ref = tau_ref(si);
end

function fit_ref = fit_local_reference(f_ref, tau_ref)
span_ref = max(f_ref) - min(f_ref);
deg = 1;
x0 = mean(f_ref);
x_scale = max(span_ref / 2, 1);
x_fit = (f_ref - x0) / x_scale;

fit_ref.deg = deg;
fit_ref.x0 = x0;
fit_ref.x_scale = x_scale;
fit_ref.p = polyfit(x_fit, tau_ref, deg);
end

function tau_pred = local_predict(f_query, f_ref, tau_ref, fit_ref)
tau_pred = interp1(f_ref, tau_ref, f_query, 'linear', NaN);
mask_out = isnan(tau_pred);
if any(mask_out)
    x_query = (f_query(mask_out) - fit_ref.x0) / fit_ref.x_scale;
    tau_pred(mask_out) = polyval(fit_ref.p, x_query);
end
end

function tau_pred = predict_target(f_query, f_ref, tau_ref, fit_ref, uplift_model)
tau_local = local_predict(f_query, f_ref, tau_ref, fit_ref);
tau_pred = tau_local;

if uplift_model.enabled
    span_edge = max(uplift_model.transition_hi - uplift_model.transition_lo, eps);
    beta = min(max((f_query - uplift_model.transition_lo) / span_edge, 0), 1);
    beta = beta .^ uplift_model.power;
    tau_uplift = tau_local + beta * uplift_model.delta_edge;
    tau_pred = max(tau_pred, tau_uplift);
end
end

function uplift_model = build_edge_uplift_model(f_ref, tau_ref, fit_ref, ...
    transition_lo, transition_hi, gain, power, uplift_cap)
uplift_model.enabled = false;
uplift_model.transition_lo = transition_lo;
uplift_model.transition_hi = transition_hi;
uplift_model.power = max(power, 0.25);
uplift_model.tau_edge_local = local_predict(transition_hi, f_ref, tau_ref, fit_ref);
uplift_model.tau_edge_target = uplift_model.tau_edge_local;
uplift_model.delta_edge = 0;

if gain <= 0
    return;
end

% 纯数据驱动：仅用右侧参考区数据估计边缘上限
tau_edge_cap = max(max(tau_ref), uplift_model.tau_edge_local);

delta_edge = max(tau_edge_cap - uplift_model.tau_edge_local, 0);
delta_edge = min(gain * delta_edge, uplift_cap);
if delta_edge <= 0
    return;
end

uplift_model.enabled = true;
uplift_model.delta_edge = delta_edge;
uplift_model.tau_edge_target = uplift_model.tau_edge_local + delta_edge;
end

function s = robust_mad_nonzero(x)
x = x(isfinite(x) & x > 0);
if isempty(x)
    s = 0;
else
    s = 1.4826 * median(abs(x - median(x)));
    if ~isfinite(s) || s <= 0
        s = 1.4826 * median(x);
    end
end
end

function value = get_cfg(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = cfg.(field_name);
else
    value = default_value;
end
end
