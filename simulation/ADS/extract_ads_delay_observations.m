%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频信号时延散点提取与盲化清洗
% 用途：统一生成第五章 5.3 所需的散点、权重与清洗诊断信息
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obs = extract_ads_delay_observations(opts)

if nargin < 1 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'enable_repair')
    opts.enable_repair = true;
end


%% 1. Load raw ADS IF signal
data = readmatrix('hunpin_time_v.txt', 'FileType', 'text', 'NumHeaderLines', 1);
valid = ~isnan(data(:, 1)) & ~isnan(data(:, 2));
t_raw = data(valid, 1);
v_raw = data(valid, 2);

T_m = t_raw(end) - t_raw(1);
f_start = 34.4e9;
f_end = 37.61e9;
K = (f_end - f_start) / T_m;
baseline_delay = 0.2470e-9;

%% 2. Resample and low-pass filter
fs_dec = 4e9;
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');
[b_lp, a_lp] = butter(4, 200e6 / (fs_dec / 2));
s_if = filtfilt(b_lp, a_lp, v_dec);

s_proc = s_if(1:2:end);
t_proc = t_dec(1:2:end);
f_s_proc = fs_dec / 2;

%% 3. Sliding-window ESPRIT extraction
win_len = max(round(0.03 * length(s_proc)), 64);
step_len = max(round(win_len / 8), 1);
L_sub = round(win_len / 2);
rms_threshold = max(abs(s_proc)) * 0.005;
edge_margin = max(0.01, 0.5 * win_len / length(s_proc));
num_windows = floor((length(s_proc) - win_len) / step_len) + 1;

f_probe = zeros(num_windows, 1);
tau_est = zeros(num_windows, 1);
amp_est = zeros(num_windows, 1);
quality_est = zeros(num_windows, 1);
count = 0;

for i = 1:num_windows
    idx = (i-1) * step_len + 1 : (i-1) * step_len + win_len;
    if idx(end) > length(s_proc)
        break;
    end

    x_win = s_proc(idx);
    t_c = t_proc(idx(round(win_len / 2)));
    t_ratio = t_c / T_m;
    if t_ratio < edge_margin || t_ratio > (1 - edge_margin) || rms(x_win) < rms_threshold
        continue;
    end

    M_sub = win_len - L_sub + 1;
    X_h = zeros(L_sub, M_sub);
    for k = 1:M_sub
        X_h(:, k) = x_win(k : k + L_sub - 1).';
    end

    R_forward = (X_h * X_h') / M_sub;
    J = fliplr(eye(L_sub));
    R_x = (R_forward + J * conj(R_forward) * J) / 2;
    [V, D] = eig(R_x);
    [lam, id] = sort(diag(D), 'descend');
    V = V(:, id);

    mdl = zeros(length(lam), 1);
    for k = 0:length(lam)-1
        noise_sv = lam(k+1:end);
        noise_sv(noise_sv < 1e-30) = 1e-30;
        mdl(k+1) = -(length(lam) - k) * M_sub * ...
            log(prod(noise_sv)^(1 / length(noise_sv)) / mean(noise_sv)) + ...
            0.5 * k * (2 * length(lam) - k) * log(M_sub);
    end
    [~, k_est] = min(mdl);
    num_s = min(max(1, k_est - 1), 3);

    Us = V(:, 1:num_s);
    Phi = (Us(1:end-1, :)' * Us(1:end-1, :)) \ ...
        (Us(1:end-1, :)' * Us(2:end, :));
    est_f = abs(angle(eig(Phi))) * f_s_proc / (2 * pi);
    est_f = est_f(est_f > 50e3 & est_f < f_s_proc / 4);
    if isempty(est_f)
        continue;
    end

    count = count + 1;
    f_probe(count) = f_start + K * t_c;
    tau_est(count) = min(est_f) / K - baseline_delay;
    amp_est(count) = rms(x_win);
    quality_est(count) = estimate_local_quality(x_win, f_s_proc);
end

f_probe = f_probe(1:count);
tau_est = tau_est(1:count);
amp_est = amp_est(1:count);
quality_est = quality_est(1:count);

%% 4. Blind cleaning: amplitude gate + bootstrap floor + continuity repair
amp_norm = amp_est / max(amp_est + eps);
mask_amp = amp_norm > 0.20;

[f_amp, sort_idx] = sort(f_probe(mask_amp));
tau_amp = tau_est(mask_amp);
tau_amp = tau_amp(sort_idx);
amp_amp = amp_est(mask_amp);
amp_amp = amp_amp(sort_idx);
amp_norm_amp = amp_norm(mask_amp);
amp_norm_amp = amp_norm_amp(sort_idx);
quality_amp = quality_est(mask_amp);
quality_amp = quality_amp(sort_idx);

core_mask = amp_norm_amp >= 0.75;
if sum(core_mask) < min(8, numel(core_mask))
    [~, amp_idx] = sort(amp_norm_amp, 'descend');
    core_mask = false(size(amp_norm_amp));
    core_mask(amp_idx(1:min(8, numel(amp_idx)))) = true;
end

tau_core = tau_amp(core_mask);
tau_q25 = prctile(tau_core, 25);
tau_q75 = prctile(tau_core, 75);
tau_floor = max(tau_q25 - 1.5 * (tau_q75 - tau_q25), 0);
mask_floor = tau_amp >= tau_floor;

local_span = min(5, make_odd(max(3, 2 * floor(numel(tau_amp) / 20) + 1)));
local_med = movmedian(tau_amp, local_span);
local_mad = movmedian(abs(tau_amp - local_med), local_span) + 1e-13;
dip_depth = local_med - tau_amp;
dip_threshold = max(3 * 1.4826 * local_mad, 0.35e-9);
quality_med = movmedian(quality_amp, local_span) + eps;
quality_ratio = quality_amp ./ quality_med;
mask_repair = mask_floor & (dip_depth > dip_threshold) & (quality_ratio < 0.97);

tau_clean = tau_amp;
for i = find(mask_repair).'
    idx_nb = max(1, i-2) : min(length(tau_amp), i+2);
    idx_nb = idx_nb(~mask_repair(idx_nb) & mask_floor(idx_nb));
    idx_nb = idx_nb(idx_nb ~= i);
    if isempty(idx_nb)
        tau_clean(i) = local_med(i);
    else
        w_nb = amp_norm_amp(idx_nb).^2;
        tau_clean(i) = sum(w_nb .* tau_amp(idx_nb)) / sum(w_nb + eps);
    end
end

mask_valid = mask_floor;
f_fit = f_amp(mask_valid);
tau_fit_pre = tau_amp(mask_valid);
tau_fit_post = tau_clean(mask_valid);
if opts.enable_repair
    tau_fit = tau_fit_post;
    cleaning_mode = 'post_repair';
else
    tau_fit = tau_fit_pre;
    cleaning_mode = 'pre_repair';
end
amp_fit = amp_amp(mask_valid);
amp_norm_fit = amp_norm_amp(mask_valid);
quality_fit = quality_amp(mask_valid);

q_p10 = prctile(quality_fit, 10);
q_p90 = prctile(quality_fit, 90);
quality_soft = (quality_fit - q_p10) ./ (q_p90 - q_p10 + eps);
quality_soft = min(max(quality_soft, 0), 1);
weights = amp_norm_fit.^2 .* (0.60 + 0.40 * quality_soft);
weights = weights / max(weights + eps);

%% 5. Package outputs
obs = struct();
obs.baseline_delay = baseline_delay;
obs.edge_margin = edge_margin;
obs.f_start = f_start;
obs.f_end = f_end;
obs.K = K;
obs.sigma_meas = 0.2e-9;

obs.f_raw = f_probe;
obs.tau_raw = tau_est;
obs.amp_raw = amp_est;
obs.quality_raw = quality_est;

obs.f_amp = f_amp;
obs.tau_amp = tau_amp;
obs.amp_amp = amp_amp;
obs.amp_norm_amp = amp_norm_amp;
obs.quality_amp = quality_amp;
obs.local_median = local_med;
obs.local_mad = local_mad;
obs.quality_ratio = quality_ratio;
obs.mask_floor = mask_floor;
obs.mask_repair = mask_repair;
obs.tau_floor = tau_floor;
obs.tau_floor_ci = [tau_q25, tau_q75];

obs.f_fit = f_fit;
obs.tau_fit = tau_fit;
obs.tau_fit_pre = tau_fit_pre;
obs.tau_fit_post = tau_fit_post;
obs.cleaning_mode = cleaning_mode;
obs.enable_repair = logical(opts.enable_repair);
obs.f_repair = f_amp(mask_repair);
obs.tau_repair_pre = tau_amp(mask_repair);
obs.tau_repair_post = tau_clean(mask_repair);
obs.repair_delta = tau_clean(mask_repair) - tau_amp(mask_repair);
obs.amp_fit = amp_fit;
obs.quality_fit = quality_fit;
obs.weights = weights(:);

obs.diag = struct();
obs.diag.raw_count = numel(f_probe);
obs.diag.amp_count = numel(f_amp);
obs.diag.floor_count = sum(mask_floor);
obs.diag.repair_count = sum(mask_repair);
obs.diag.final_count = numel(f_fit);
obs.diag.rms_threshold = rms_threshold;
obs.diag.local_span = local_span;
end

function quality_ratio = estimate_local_quality(x_win, fs_proc)
nfft_local = 2^nextpow2(max(length(x_win), 256));
xw = x_win(:) .* hann(length(x_win));
S_local = abs(fft(xw, nfft_local));
f_axis = (0:nfft_local-1).' * (fs_proc / nfft_local);
band_mask = (f_axis > 50e3) & (f_axis < fs_proc / 4);
if any(band_mask)
    S_band = S_local(band_mask);
    quality_ratio = max(S_band) / (median(S_band) + eps);
else
    quality_ratio = 1;
end
end

function val = make_odd(val)
if mod(val, 2) == 0
    val = val + 1;
end
end
