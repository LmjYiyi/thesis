function out = rebuild_right_edge(hybrid_cal, v_proc, t_proc, fs_proc, ...
    f_start, K, rms_thr, base_cal, cfg, f_valid_lo, f_beat_max, show_summary)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 右侧边缘镜像重建
% 基于通带对称性假设，将左侧轨迹镜像到右侧作为参考，
% 多窗口 + 多 L_sub 提取后以镜像引导选模
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

esp = esprit_extract();

% 1. 构建镜像参考（清洗左侧离群点后镜像）
f_center = 37.0e9;
mask_left = hybrid_cal.f_probe < f_center;

if sum(mask_left) >= 5
    [f_left, si] = sort(hybrid_cal.f_probe(mask_left));
    tau_left = hybrid_cal.tau(mask_left); tau_left = tau_left(si);

    % 清除左侧离群
    if numel(tau_left) >= 5
        sp = max(5, 2*floor(numel(tau_left)/8)+1);
        if mod(sp,2)==0, sp = sp+1; end
        tlm = movmedian(tau_left, sp);
        dev = abs(tau_left - tlm);
        thr = max(3*1.4826*movmedian(dev, sp), 0.2e-9);
        ok = dev <= thr;
        f_left = f_left(ok); tau_left = tau_left(ok);
    end
    f_ref = 2*f_center - flipud(f_left);
    tau_ref = flipud(tau_left);
else
    mr = hybrid_cal.f_probe >= (cfg.band_lo-0.10e9) & ...
         hybrid_cal.f_probe <= cfg.band_hi;
    [f_ref, si] = sort(hybrid_cal.f_probe(mr));
    tau_ref = hybrid_cal.tau(mr); tau_ref = tau_ref(si);
end

[f_ref, ui] = unique(f_ref, 'stable');
tau_ref = tau_ref(ui);
if numel(tau_ref) >= 5
    sp = max(3, 2*floor(numel(tau_ref)/12)+1);
    if mod(sp,2)==0, sp = sp+1; end
    tau_ref = movmedian(tau_ref, sp);
end

if show_summary
    fprintf('  [镜像参考] f = %.3f-%.3f GHz, tau = %.2f-%.2f ns\n', ...
        min(f_ref)/1e9, max(f_ref)/1e9, min(tau_ref)*1e9, max(tau_ref)*1e9);
end

% 2. 清洗旧右边缘点
purge_lo = cfg.band_lo;
if isfield(cfg, 'purge_band_lo'), purge_lo = cfg.purge_band_lo; end
mask_in = hybrid_cal.f_probe >= purge_lo & hybrid_cal.f_probe <= cfg.band_hi;
mask_keep = true(numel(hybrid_cal.f_probe), 1);

if any(mask_in)
    tp = interp1(f_ref, tau_ref, hybrid_cal.f_probe(mask_in), 'linear', 'extrap');
    bad = abs(hybrid_cal.tau(mask_in) - tp) > cfg.purge_tol;
    idx_in = find(mask_in);
    mask_keep(idx_in(bad)) = false;
end

n_purged = sum(~mask_keep);
f_kept = hybrid_cal.f_probe(mask_keep); tau_kept = hybrid_cal.tau(mask_keep);
amp_kept = hybrid_cal.amp(mask_keep);   win_kept = hybrid_cal.win_len(mask_keep);
src_kept = hybrid_cal.source_code(mask_keep);

if show_summary
    fprintf('  [清洗旧点] 频段内 %d 个, 剔除 %d 个\n', sum(mask_in), n_purged);
end

% 3. 多窗口 + 多 L_sub 提取，镜像引导选模
f_all = []; tau_all = []; amp_all = []; win_all = [];
N_proc = length(v_proc);
L_sub_ratios = [1/2];
if isfield(cfg, 'L_sub_ratios'), L_sub_ratios = cfg.L_sub_ratios; end

for iw = 1:numel(cfg.win_lens)
    wl = cfg.win_lens(iw);
    for ir = 1:numel(L_sub_ratios)
        Ls = max(4, round(wl * L_sub_ratios(ir)));
        centers = (round(wl/2)+1 : cfg.step_len : N_proc-round(wl/2));
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

            % 选最接近镜像参考的模态
            te = interp1(f_ref, tau_ref, fc, 'linear', 'extrap');
            [~, bi] = min(abs(ta - te));

            f_all = [f_all; fc]; tau_all = [tau_all; ta(bi)];
            amp_all = [amp_all; av]; win_all = [win_all; wl];
            nc = nc + 1;
        end

        if show_summary
            fprintf('  win=%d, L_sub=%d (%.0f%%): %d 个\n', wl, Ls, L_sub_ratios(ir)*100, nc);
        end
    end
end

% 4. 同频去重：选最接近参考的
if numel(f_all) > 1
    tp_all = interp1(f_ref, tau_ref, f_all, 'linear', 'extrap');
    dv = abs(tau_all - tp_all);
    [f_all, si] = sort(f_all);
    tau_all = tau_all(si); amp_all = amp_all(si);
    win_all = win_all(si); dv = dv(si);

    keep = true(numel(f_all), 1);
    ii = 1;
    while ii <= numel(f_all)
        jj = ii;
        while jj < numel(f_all) && (f_all(jj+1)-f_all(ii)) < 0.004e9
            jj = jj + 1;
        end
        if jj > ii
            g = ii:jj;
            [~, best] = min(dv(g));
            mk = false(numel(g),1); mk(best) = true;
            keep(g(~mk)) = false;
        end
        ii = jj + 1;
    end
    f_all = f_all(keep); tau_all = tau_all(keep);
    amp_all = amp_all(keep); win_all = win_all(keep);
end

% 5. 一致性筛选
n_candidates = numel(f_all);
if ~isempty(f_all)
    tp = interp1(f_ref, tau_ref, f_all, 'linear', 'extrap');
    mc = (tau_all >= tp - cfg.tau_tol_lo) & (tau_all <= tp + cfg.tau_tol_hi);
    f_all = f_all(mc); tau_all = tau_all(mc);
    amp_all = amp_all(mc); win_all = win_all(mc);
end

% 6. 合并
is_new = false(size(f_all));
for i = 1:numel(f_all)
    if isempty(f_kept) || min(abs(f_kept - f_all(i))) > cfg.min_freq_gap
        is_new(i) = true;
    end
end

f_m = [f_kept; f_all(is_new)]; tau_m = [tau_kept; tau_all(is_new)];
amp_m = [amp_kept; amp_all(is_new)]; win_m = [win_kept; win_all(is_new)];
src_m = [src_kept; 3*ones(sum(is_new),1)];

[f_m, si] = sort(f_m);
tau_m = tau_m(si); amp_m = amp_m(si); win_m = win_m(si); src_m = src_m(si);

out.f_probe = f_m; out.tau = tau_m; out.amp = amp_m;
out.win_len = win_m; out.source_code = src_m;

if show_summary
    fprintf('  重建: 剔除 %d, 候选 %d, 一致性保留 %d, 新增 %d, 总计 %d\n', ...
        n_purged, n_candidates, numel(f_all), sum(is_new), numel(out.f_probe));
end
end
