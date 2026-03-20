function fn = esprit_extract()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ESPRIT extraction helpers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fn.core           = @esprit_core;
fn.single_window  = @process_single_window;
fn.all_modes      = @process_window_all_modes;
fn.run_fixed      = @run_fixed_extraction;
fn.run_adaptive   = @run_adaptive_extraction;
end

%% ---- ESPRIT core: MDL + ESPRIT ----
function [est_f, proj_power] = esprit_core(x_win, fs_proc, L_sub, f_valid_lo)
est_f = [];
proj_power = [];

win_len = numel(x_win);
M_sub = win_len - L_sub + 1;
if M_sub <= 2, return; end

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

mdl_v = zeros(length(lam), 1);
for kk = 0:length(lam)-1
    ns = lam(kk + 1:end);
    ns(ns < 1e-30) = 1e-30;
    mdl_v(kk + 1) = -(length(lam) - kk) * M_sub * ...
        log(prod(ns)^(1 / length(ns)) / mean(ns)) + ...
        0.5 * kk * (2 * length(lam) - kk) * log(M_sub);
end
[
~, k_est] = min(mdl_v);
num_s = min(max(1, k_est - 1), 3);

Us = V(:, 1:num_s);
Phi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ ...
      (Us(1:end-1, :)' * Us(2:end, :));
eig_vals = eig(Phi);
est_f = abs(angle(eig_vals)) * fs_proc / (2 * pi);

valid_mask = est_f > f_valid_lo & est_f < fs_proc / 4;
est_f = est_f(valid_mask);
if isempty(est_f), return; end

proj_power = zeros(numel(est_f), 1);
for jj = 1:numel(est_f)
    steering = exp(1j * 2 * pi * est_f(jj) / fs_proc * (0:L_sub-1).');
    proj_power(jj) = abs(steering' * V(:, 1))^2;
end
end

%% ---- single-window extraction: strongest valid mode ----
function [is_ok, f_probe, tau_val, amp_val, center_idx, reason_code] = ...
    process_single_window(x_win, idx, t_proc, fs_proc, f_start, K, ...
    rms_thr, L_sub, f_valid_lo, f_beat_max)

is_ok = false;
f_probe = NaN;
tau_val = NaN;
amp_val = rms(x_win);
center_idx = idx(round(numel(idx) / 2));
reason_code = 0;

if amp_val < rms_thr
    reason_code = 1;
    return;
end

[est_f, proj_power] = esprit_core(x_win, fs_proc, L_sub, f_valid_lo);
if isempty(est_f)
    reason_code = 2;
    return;
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

%% ---- single-window extraction: return all valid modes ----
function [f_probe, tau_all, amp_val, center_idx, proj_power] = ...
    process_window_all_modes(x_win, idx, t_proc, fs_proc, f_start, K, ...
    rms_thr, L_sub, f_valid_lo, f_beat_max)

f_probe = NaN;
tau_all = [];
amp_val = 0;
proj_power = [];
center_idx = idx(round(numel(idx) / 2));
amp_val = rms(x_win);

if amp_val < rms_thr, return; end

[est_f, pp] = esprit_core(x_win, fs_proc, L_sub, f_valid_lo);
if isempty(est_f), return; end

mask_valid = est_f <= f_beat_max;
est_f = est_f(mask_valid);
pp = pp(mask_valid);
if isempty(est_f), return; end

t_c = t_proc(center_idx);
f_probe = f_start + K * t_c;
tau_all = est_f / K;
proj_power = pp;
end

%% ---- fixed-window extraction ----
function out = run_fixed_extraction(v_proc, t_proc, fs_proc, f_start, K, ...
    rms_thr, cfg, f_valid_lo, f_beat_max, show_summary)

N_proc  = length(v_proc);
num_win = floor((N_proc - cfg.win_len) / cfg.step_len) + 1;

f_arr   = zeros(num_win, 1);
tau_arr = zeros(num_win, 1);
amp_arr = zeros(num_win, 1);
ctr_arr = zeros(num_win, 1);
win_arr = zeros(num_win, 1);
cnt = 0;
n_skip = [0 0 0];

for i = 1:num_win
    idx = (i-1)*cfg.step_len + 1 : (i-1)*cfg.step_len + cfg.win_len;
    if idx(end) > N_proc, break; end

    [ok, fp, tv, av, ci, rc] = process_single_window( ...
        v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
        rms_thr, cfg.L_sub, f_valid_lo, f_beat_max);

    if ~ok
        if rc >= 1 && rc <= 3, n_skip(rc) = n_skip(rc) + 1; end
        continue;
    end

    cnt = cnt + 1;
    f_arr(cnt) = fp;
    tau_arr(cnt) = tv;
    amp_arr(cnt) = av;
    ctr_arr(cnt) = ci;
    win_arr(cnt) = cfg.win_len;
end

out.f_probe    = f_arr(1:cnt);
out.tau        = tau_arr(1:cnt);
out.amp        = amp_arr(1:cnt);
out.center_idx = ctr_arr(1:cnt);
out.win_len    = win_arr(1:cnt);

if show_summary
    fprintf('  %s: win=%d (%.1f us), step=%d, L_sub=%d, 共 %d 窗口\n', ...
        cfg.name, cfg.win_len, cfg.win_len/fs_proc*1e6, cfg.step_len, cfg.L_sub, num_win);
    fprintf('  原始散点: %d / %d, 跳过: RMS=%d, 无效频率=%d, 超上界=%d\n', ...
        cnt, num_win, n_skip(1), n_skip(2), n_skip(3));
end
end

%% ---- adaptive-window extraction ----
function out = run_adaptive_extraction(v_proc, t_proc, fs_proc, f_start, K, ...
    rms_thr, base_clean, cfg, f_valid_lo, f_beat_max)

N_proc = length(v_proc);
center_grid = (round(cfg.win_short/2)+1 : cfg.step_center : ...
    N_proc - round(cfg.win_short/2)).';

[f_base_sort, si] = sort(base_clean.f_probe);
tau_base_sort = base_clean.tau(si);
span_s = max(5, 2*floor(numel(tau_base_sort)/12)+1);
if mod(span_s,2)==0, span_s = span_s+1; end
tau_smooth = movmean(tau_base_sort, span_s);

f_grid = f_start + K * t_proc(center_grid);
tau_pred = interp1(f_base_sort, tau_smooth, f_grid, 'linear', 'extrap');

N_grid = numel(center_grid);
f_arr = zeros(N_grid,1);
tau_arr = zeros(N_grid,1);
amp_arr = zeros(N_grid,1);
ctr_arr = zeros(N_grid,1);
win_arr = zeros(N_grid,1);
tpred_arr = zeros(N_grid,1);
cnt = 0;
n_win = [0 0 0 0];

for i = 1:N_grid
    ci = center_grid(i);
    tn = tau_pred(i);

    if     tn < cfg.tau_thr_1, wl = cfg.win_long;  n_win(4) = n_win(4)+1;
    elseif tn < cfg.tau_thr_2, wl = cfg.win_mid2;  n_win(3) = n_win(3)+1;
    elseif tn < cfg.tau_thr_3, wl = cfg.win_mid1;  n_win(2) = n_win(2)+1;
    else,                      wl = cfg.win_short; n_win(1) = n_win(1)+1;
    end

    L_sub = round(wl / 2);
    idx_s = max(1, min(ci - floor(wl/2), N_proc - wl + 1));
    idx = idx_s : idx_s + wl - 1;

    [ok, fp, tv, av, cn, rc] = process_single_window( ...
        v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
        rms_thr, L_sub, f_valid_lo, f_beat_max);

    if ~ok && rc == 2 && wl < cfg.win_long
        wl2 = min(cfg.win_long, N_proc);
        idx_s = max(1, min(ci - floor(wl2/2), N_proc - wl2 + 1));
        idx = idx_s : idx_s + wl2 - 1;
        L_sub = round(wl2 / 2);
        [ok, fp, tv, av, cn, ~] = process_single_window( ...
            v_proc(idx), idx, t_proc, fs_proc, f_start, K, ...
            rms_thr, L_sub, f_valid_lo, f_beat_max);
        if ok, wl = wl2; end
    end

    if ~ok, continue; end

    cnt = cnt + 1;
    f_arr(cnt) = fp;
    tau_arr(cnt) = tv;
    amp_arr(cnt) = av;
    ctr_arr(cnt) = cn;
    win_arr(cnt) = wl;
    tpred_arr(cnt) = tn;
end

[ctr_unique, ia] = unique(ctr_arr(1:cnt), 'stable');
out.f_probe    = f_arr(ia);
out.tau        = tau_arr(ia);
out.amp        = amp_arr(ia);
out.center_idx = ctr_unique;
out.win_len    = win_arr(ia);
out.tau_pred   = tpred_arr(ia);

fprintf('  自适应中心网格: %d 个\n', N_grid);
fprintf('  窗口分配: short=%d, mid1=%d, mid2=%d, long=%d\n', ...
    n_win(1), n_win(2), n_win(3), n_win(4));
fprintf('  自适应原始散点: %d / %d\n', numel(out.f_probe), N_grid);
end
