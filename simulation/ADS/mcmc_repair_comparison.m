%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS????????????? MCMC ??
% ??????? pre-repair ? post-repair ??????5-5?MCMC?????
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
rng(42);

fprintf('======================================================\n');
fprintf('  ADS?????? pre/post repair MCMC ??\n');
fprintf('======================================================\n\n');

config = struct();
config.n_chains = 4;
config.n_samples = 18000;
config.burn_in = 6000;
config.adapt_end = config.burn_in;
config.adapt_window = 200;
config.target_accept = 0.30;
config.initial_sigma = [0.006e9, 0.010e9, 0.10];
config.bounds = [36.0e9, 38.0e9; 0.5e9, 2.0e9; 2.0, 8.0];
config.param_names = {'F0', 'BW', 'N'};
config.rng_seed = 42;

obs_pre = extract_ads_delay_observations(struct('enable_repair', false));
obs_post = extract_ads_delay_observations(struct('enable_repair', true));

fprintf('Observation count: pre %d, post %d\n', numel(obs_pre.f_fit), numel(obs_post.f_fit));
fprintf('Repair count: %d\n', obs_post.diag.repair_count);
for k = 1:numel(obs_post.f_repair)
    fprintf('  Repair %d: f=%.6f GHz, pre=%.4f ns, post=%.4f ns, delta=%.4f ns\n', ...
        k, obs_post.f_repair(k)/1e9, obs_post.tau_repair_pre(k)*1e9, ...
        obs_post.tau_repair_post(k)*1e9, obs_post.repair_delta(k)*1e9);
end
fprintf('\n');

stats_pre = compute_scatter_stats(obs_pre);
stats_post = compute_scatter_stats(obs_post);
summary_pre = run_mcmc_case(obs_pre, config, 'pre_repair', false);
summary_post = run_mcmc_case(obs_post, config, 'post_repair', false);

comparison = struct();
comparison.config = config;
comparison.obs_pre = obs_pre;
comparison.obs_post = obs_post;
comparison.stats_pre = stats_pre;
comparison.stats_post = stats_post;
comparison.summary_pre = summary_pre;
comparison.summary_post = summary_post;
comparison.posterior_mean_delta = summary_post.posterior_mean - summary_pre.posterior_mean;
comparison.posterior_map_delta = summary_post.posterior_map - summary_pre.posterior_map;
comparison.posterior_std_delta = summary_post.posterior_std - summary_pre.posterior_std;
comparison.cv_delta = summary_post.cv - summary_pre.cv;
comparison.rhat_delta = summary_post.posterior_rhat - summary_pre.posterior_rhat;
comparison.ess_delta = summary_post.posterior_ess - summary_pre.posterior_ess;
comparison.iact_delta = summary_post.posterior_iact - summary_pre.posterior_iact;

save('mcmc_repair_comparison.mat', 'comparison');
write_comparison_text('mcmc_repair_comparison_summary.txt', comparison);

fprintf('Comparison written to: %s\n', fullfile(pwd, 'mcmc_repair_comparison_summary.txt'));
fprintf('MAT file written to: %s\n', fullfile(pwd, 'mcmc_repair_comparison.mat'));
fprintf('======================================================\n');

function summary = run_mcmc_case(obs, config, case_label, export_figures)
rng(config.rng_seed);

fprintf('[%s] Valid observation count: %d\n', case_label, numel(obs.f_fit));
fprintf('[%s] Frequency range: %.3f - %.3f GHz\n', case_label, min(obs.f_fit)/1e9, max(obs.f_fit)/1e9);
fprintf('[%s] Delay range: %.3f - %.3f ns\n', case_label, min(obs.tau_fit)*1e9, max(obs.tau_fit)*1e9);
fprintf('[%s] Cleaning summary: raw %d -> amp %d -> floor %d -> final %d (repair %d, mode %s)\n', ...
    case_label, obs.diag.raw_count, obs.diag.amp_count, obs.diag.floor_count, ...
    obs.diag.final_count, obs.diag.repair_count, obs.cleaning_mode);

n_params = size(config.bounds, 1);
base_state = estimate_initial_state(obs.f_fit, obs.tau_fit, config.bounds);
init_offsets = [ ...
    -0.10e9, -0.08e9, -0.8; ...
     0.10e9,  0.08e9,  0.8; ...
    -0.06e9,  0.12e9,  0.4; ...
     0.06e9, -0.12e9, -0.4];
init_states = zeros(config.n_chains, n_params);
for c = 1:config.n_chains
    init_states(c, :) = clip_to_bounds(base_state + init_offsets(c, :), config.bounds);
end

chains = zeros(config.n_samples, n_params, config.n_chains);
logL_chain = zeros(config.n_samples, config.n_chains);
accept_rate = zeros(n_params, config.n_chains);
final_sigma = zeros(n_params, config.n_chains);
init_logL = zeros(config.n_chains, 1);

for c = 1:config.n_chains
    fprintf('[%s] Chain %d/%d start...\n', case_label, c, config.n_chains);
    [chains(:, :, c), logL_chain(:, c), accept_rate(:, c), final_sigma(:, c), init_logL(c)] = ...
        run_adaptive_chain(obs, init_states(c, :), config.bounds, config.initial_sigma, config.n_samples, ...
            config.adapt_end, config.adapt_window, config.target_accept, c);
    fprintf('[%s] Chain %d accept rate: F0 %.2f%%, BW %.2f%%, N %.2f%%\n', ...
        case_label, c, 100*accept_rate(1, c), 100*accept_rate(2, c), 100*accept_rate(3, c));
end
fprintf('\n');

post_chains = chains(config.burn_in+1:end, :, :);
n_keep = size(post_chains, 1);
combined = flatten_post_chains(post_chains);
posterior_mean = mean(combined, 1);
posterior_std = std(combined, 0, 1);
posterior_ci = zeros(n_params, 2);
posterior_map = zeros(1, n_params);
posterior_rhat = zeros(1, n_params);
posterior_ess = zeros(1, n_params);
posterior_iact = zeros(1, n_params);

flat_logL = reshape(logL_chain(config.burn_in+1:end, :), [], 1);
[~, best_idx] = max(flat_logL);
posterior_map(:) = combined(best_idx, :);

fprintf('[%s] Computing Rhat, ESS, and IACT...\n', case_label);
for p = 1:n_params
    posterior_ci(p, :) = prctile(combined(:, p), [2.5, 97.5]);
    posterior_rhat(p) = compute_rhat(squeeze(post_chains(:, p, :)));
    [posterior_ess(p), posterior_iact(p)] = compute_ess_and_iact(squeeze(post_chains(:, p, :)));
end

cv = posterior_std ./ posterior_mean;
overall_accept = mean(accept_rate, 2);

summary = struct();
summary.case_label = case_label;
summary.enable_repair = obs.enable_repair;
summary.cleaning_mode = obs.cleaning_mode;
summary.n_chains = config.n_chains;
summary.n_samples = config.n_samples;
summary.burn_in = config.burn_in;
summary.n_keep_per_chain = n_keep;
summary.valid_points = numel(obs.f_fit);
summary.accept_rate = accept_rate;
summary.final_sigma = final_sigma;
summary.posterior_mean = posterior_mean;
summary.posterior_std = posterior_std;
summary.posterior_ci = posterior_ci;
summary.posterior_map = posterior_map;
summary.posterior_rhat = posterior_rhat;
summary.posterior_ess = posterior_ess;
summary.posterior_iact = posterior_iact;
summary.cv = cv;
summary.corr_matrix = corrcoef(combined);
summary.weight_range = [min(obs.weights), max(obs.weights)];
summary.bootstrap_tau_floor = obs.tau_floor;
summary.edge_margin = obs.edge_margin;
summary.cleaning_counts = [obs.diag.raw_count, obs.diag.amp_count, obs.diag.floor_count, obs.diag.final_count];
summary.logL_max = max(flat_logL);
summary.logL_mean = mean(flat_logL);
summary.logL_std = std(flat_logL);
summary.overall_accept = overall_accept;
summary.init_logL = init_logL;
summary.repair_count = obs.diag.repair_count;
summary.repair_delta_ns = obs.repair_delta(:).' * 1e9;

write_summary_text(sprintf('mcmc_%s_summary.txt', case_label), summary);
save(sprintf('mcmc_%s_summary.mat', case_label), 'summary');

fprintf('[%s] Posterior mean: F0 %.4f GHz, BW %.4f GHz, N %.4f\n', ...
    case_label, posterior_mean(1)/1e9, posterior_mean(2)/1e9, posterior_mean(3));
fprintf('[%s] Rhat: F0 %.4f, BW %.4f, N %.4f\n', ...
    case_label, posterior_rhat(1), posterior_rhat(2), posterior_rhat(3));
fprintf('[%s] ESS: F0 %.1f, BW %.1f, N %.1f\n\n', ...
    case_label, posterior_ess(1), posterior_ess(2), posterior_ess(3));

if export_figures
    figure_dir = fullfile(pwd, '..', '..', 'output', 'figures');
    if ~exist(figure_dir, 'dir')
        mkdir(figure_dir);
    end
    plot_trace_and_marginals(chains, config.burn_in, config.param_names, posterior_mean, posterior_ci, ...
        figure_dir, sprintf('%s_trace.png', case_label));
    plot_corner(post_chains, posterior_mean, config.param_names, figure_dir, sprintf('%s_corner.png', case_label));
    plot_posterior_reconstruction(obs, combined, posterior_mean, figure_dir, sprintf('%s_reconstruction.png', case_label));
end
end

function stats = compute_scatter_stats(obs)
delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
f_true = delay_data(:, 1);
tau_true = delay_data(:, 2);
tau_true_at_scatter = interp1(f_true, tau_true, obs.f_fit, 'pchip');
residuals = obs.tau_fit - tau_true_at_scatter;
abs_residuals = abs(residuals);
f_ghz = obs.f_fit / 1e9;

mask_flat = (f_ghz >= 36.7) & (f_ghz <= 37.3);
mask_transition = ((f_ghz >= 36.5) & (f_ghz < 36.7)) | ((f_ghz > 37.3) & (f_ghz <= 37.5));
mask_peak = ((f_ghz >= 36.43) & (f_ghz < 36.5)) | (f_ghz > 37.5);
mask_unclassified = ~mask_flat & ~mask_transition & ~mask_peak;
if any(mask_unclassified)
    mask_transition = mask_transition | mask_unclassified;
end

stats = struct();
stats.flat = zone_stats(residuals, abs_residuals, mask_flat);
stats.transition = zone_stats(residuals, abs_residuals, mask_transition);
stats.peak = zone_stats(residuals, abs_residuals, mask_peak);
stats.all = zone_stats(residuals, abs_residuals, true(size(residuals)));
end

function out = zone_stats(residuals, abs_residuals, mask)
out = struct('count', sum(mask), 'mae', NaN, 'rmse', NaN, 'max_dev', NaN, 'bias', NaN);
if ~any(mask)
    return;
end
res_zone = residuals(mask);
abs_zone = abs_residuals(mask);
out.mae = mean(abs_zone);
out.rmse = sqrt(mean(res_zone.^2));
out.max_dev = max(abs_zone);
out.bias = mean(res_zone);
end

function write_comparison_text(file_path, comparison)
fid = fopen(file_path, 'w');
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Pre/post repair MCMC comparison\n');
fprintf(fid, 'repair_count=%d\n', comparison.obs_post.diag.repair_count);
for k = 1:numel(comparison.obs_post.f_repair)
    fprintf(fid, 'repair_%d_f_GHz=%.6f\n', k, comparison.obs_post.f_repair(k)/1e9);
    fprintf(fid, 'repair_%d_tau_pre_ns=%.6f\n', k, comparison.obs_post.tau_repair_pre(k)*1e9);
    fprintf(fid, 'repair_%d_tau_post_ns=%.6f\n', k, comparison.obs_post.tau_repair_post(k)*1e9);
    fprintf(fid, 'repair_%d_delta_ns=%.6f\n', k, comparison.obs_post.repair_delta(k)*1e9);
end

write_stats_block(fid, 'flat', comparison.stats_pre.flat, comparison.stats_post.flat);
write_stats_block(fid, 'transition', comparison.stats_pre.transition, comparison.stats_post.transition);
write_stats_block(fid, 'peak', comparison.stats_pre.peak, comparison.stats_post.peak);
write_stats_block(fid, 'all', comparison.stats_pre.all, comparison.stats_post.all);

fprintf(fid, '\nPosterior_mean_pre\n');
fprintf(fid, 'F0_GHz=%.6f\n', comparison.summary_pre.posterior_mean(1)/1e9);
fprintf(fid, 'BW_GHz=%.6f\n', comparison.summary_pre.posterior_mean(2)/1e9);
fprintf(fid, 'N=%.6f\n', comparison.summary_pre.posterior_mean(3));

fprintf(fid, '\nPosterior_mean_post\n');
fprintf(fid, 'F0_GHz=%.6f\n', comparison.summary_post.posterior_mean(1)/1e9);
fprintf(fid, 'BW_GHz=%.6f\n', comparison.summary_post.posterior_mean(2)/1e9);
fprintf(fid, 'N=%.6f\n', comparison.summary_post.posterior_mean(3));

fprintf(fid, '\nPosterior_mean_delta_post_minus_pre\n');
fprintf(fid, 'F0_MHz=%.6f\n', comparison.posterior_mean_delta(1)/1e6);
fprintf(fid, 'BW_MHz=%.6f\n', comparison.posterior_mean_delta(2)/1e6);
fprintf(fid, 'N=%.6f\n', comparison.posterior_mean_delta(3));

fprintf(fid, '\nPosterior_map_delta_post_minus_pre\n');
fprintf(fid, 'F0_MHz=%.6f\n', comparison.posterior_map_delta(1)/1e6);
fprintf(fid, 'BW_MHz=%.6f\n', comparison.posterior_map_delta(2)/1e6);
fprintf(fid, 'N=%.6f\n', comparison.posterior_map_delta(3));

fprintf(fid, '\nCV_percent_pre\n');
fprintf(fid, 'F0=%.6f\n', 100 * comparison.summary_pre.cv(1));
fprintf(fid, 'BW=%.6f\n', 100 * comparison.summary_pre.cv(2));
fprintf(fid, 'N=%.6f\n', 100 * comparison.summary_pre.cv(3));

fprintf(fid, '\nCV_percent_post\n');
fprintf(fid, 'F0=%.6f\n', 100 * comparison.summary_post.cv(1));
fprintf(fid, 'BW=%.6f\n', 100 * comparison.summary_post.cv(2));
fprintf(fid, 'N=%.6f\n', 100 * comparison.summary_post.cv(3));

fprintf(fid, '\nRhat_pre\n');
fprintf(fid, 'F0=%.6f\n', comparison.summary_pre.posterior_rhat(1));
fprintf(fid, 'BW=%.6f\n', comparison.summary_pre.posterior_rhat(2));
fprintf(fid, 'N=%.6f\n', comparison.summary_pre.posterior_rhat(3));

fprintf(fid, '\nRhat_post\n');
fprintf(fid, 'F0=%.6f\n', comparison.summary_post.posterior_rhat(1));
fprintf(fid, 'BW=%.6f\n', comparison.summary_post.posterior_rhat(2));
fprintf(fid, 'N=%.6f\n', comparison.summary_post.posterior_rhat(3));

fprintf(fid, '\nESS_pre\n');
fprintf(fid, 'F0=%.2f\n', comparison.summary_pre.posterior_ess(1));
fprintf(fid, 'BW=%.2f\n', comparison.summary_pre.posterior_ess(2));
fprintf(fid, 'N=%.2f\n', comparison.summary_pre.posterior_ess(3));

fprintf(fid, '\nESS_post\n');
fprintf(fid, 'F0=%.2f\n', comparison.summary_post.posterior_ess(1));
fprintf(fid, 'BW=%.2f\n', comparison.summary_post.posterior_ess(2));
fprintf(fid, 'N=%.2f\n', comparison.summary_post.posterior_ess(3));
end

function write_stats_block(fid, name, pre_stats, post_stats)
fprintf(fid, '\n%s_pre\n', name);
fprintf(fid, 'count=%d\n', pre_stats.count);
fprintf(fid, 'MAE_ns=%.6f\n', pre_stats.mae * 1e9);
fprintf(fid, 'RMSE_ns=%.6f\n', pre_stats.rmse * 1e9);
fprintf(fid, 'MaxDev_ns=%.6f\n', pre_stats.max_dev * 1e9);
fprintf(fid, 'Bias_ns=%.6f\n', pre_stats.bias * 1e9);

fprintf(fid, '\n%s_post\n', name);
fprintf(fid, 'count=%d\n', post_stats.count);
fprintf(fid, 'MAE_ns=%.6f\n', post_stats.mae * 1e9);
fprintf(fid, 'RMSE_ns=%.6f\n', post_stats.rmse * 1e9);
fprintf(fid, 'MaxDev_ns=%.6f\n', post_stats.max_dev * 1e9);
fprintf(fid, 'Bias_ns=%.6f\n', post_stats.bias * 1e9);

fprintf(fid, '\n%s_delta_post_minus_pre\n', name);
fprintf(fid, 'MAE_ns=%.6f\n', (post_stats.mae - pre_stats.mae) * 1e9);
fprintf(fid, 'RMSE_ns=%.6f\n', (post_stats.rmse - pre_stats.rmse) * 1e9);
fprintf(fid, 'MaxDev_ns=%.6f\n', (post_stats.max_dev - pre_stats.max_dev) * 1e9);
fprintf(fid, 'Bias_ns=%.6f\n', (post_stats.bias - pre_stats.bias) * 1e9);
end

function base_state = estimate_initial_state(f_fit, tau_fit, bounds)
[~, idx_peak_sorted] = sort(tau_fit, 'descend');
f_peak_candidates = sort(f_fit(idx_peak_sorted(1:min(8, numel(idx_peak_sorted)))));
f_split = median(f_fit);

left_candidates = f_peak_candidates(f_peak_candidates < f_split);
right_candidates = f_peak_candidates(f_peak_candidates >= f_split);
if isempty(left_candidates)
    left_candidates = min(f_fit);
end
if isempty(right_candidates)
    right_candidates = max(f_fit);
end

f_left = median(left_candidates);
f_right = median(right_candidates);
f_center = (f_left + f_right) / 2;
bw_init = max(f_right - f_left, 0.6e9);
n_init = 5.0;

base_state = [f_center, bw_init, n_init];
base_state = clip_to_bounds(base_state, bounds);
end

function [chain, logL_trace, accept_rate, sigma_final, init_logL] = run_adaptive_chain(obs, init_state, bounds, initial_sigma, n_samples, adapt_end, adapt_window, target_accept, chain_id)
n_params = size(bounds, 1);
chain = zeros(n_samples, n_params);
logL_trace = zeros(n_samples, 1);
sigma = initial_sigma(:);
accept_count = zeros(n_params, 1);
window_accept = zeros(n_params, 1);

current = init_state(:);
current_logL = compute_log_likelihood(obs.f_fit, obs.tau_fit, obs.weights, current.', obs.sigma_meas);
init_logL = current_logL;

for i = 1:n_samples
    for p = 1:n_params
        proposal = current;
        proposal(p) = current(p) + sigma(p) * randn();
        if proposal(p) < bounds(p, 1) || proposal(p) > bounds(p, 2)
            continue;
        end

        proposal_logL = compute_log_likelihood(obs.f_fit, obs.tau_fit, obs.weights, proposal.', obs.sigma_meas);
        if log(rand()) < proposal_logL - current_logL
            current = proposal;
            current_logL = proposal_logL;
            accept_count(p) = accept_count(p) + 1;
            window_accept(p) = window_accept(p) + 1;
        end
    end

    chain(i, :) = current;
    logL_trace(i) = current_logL;

    if i <= adapt_end && mod(i, adapt_window) == 0
        gamma = 0.8 / sqrt(i / adapt_window + 2);
        local_accept = window_accept / adapt_window;
        sigma = sigma .* exp(gamma * (local_accept - target_accept));

        width = bounds(:, 2) - bounds(:, 1);
        sigma_min = width * 1e-5;
        sigma_max = width * 0.20;
        sigma = min(max(sigma, sigma_min), sigma_max);
        window_accept(:) = 0;
    end

    if mod(i, 3000) == 0
        fprintf('  Chain %d progress %5.1f%%\n', chain_id, 100 * i / n_samples);
    end
end

accept_rate = accept_count / n_samples;
sigma_final = sigma;
end

function x = clip_to_bounds(x, bounds)
for p = 1:numel(x)
    x(p) = min(max(x(p), bounds(p, 1)), bounds(p, 2));
end
end

function tau_g = calculate_analytic_group_delay(f_vec, F0, BW, N)
N_int = round(N);
if N_int < 1
    N_int = 1;
end

Ripple = 0.5;
W1 = 2 * pi * (F0 - BW/2);
W2 = 2 * pi * (F0 + BW/2);
if W1 >= W2
    tau_g = zeros(size(f_vec));
    return;
end

try
    [b, a] = cheby1(N_int, Ripple, [W1, W2], 'bandpass', 's');
    w_vec = 2 * pi * f_vec;
    H = freqs(b, a, w_vec);
    phase = unwrap(angle(H));
    tau_g = -gradient(phase) ./ gradient(w_vec);
    tau_g(tau_g < 0) = 0;
catch
    tau_g = zeros(size(f_vec));
end
end

function logL = compute_log_likelihood(f_data, tau_data, weights, theta, sigma)
try
    if any(~isfinite(theta))
        logL = -1e12;
        return;
    end
    tau_theory = calculate_analytic_group_delay(f_data, theta(1), theta(2), theta(3));
    residuals = (tau_theory(:) - tau_data(:)) / sigma;
    logL = -0.5 * sum(weights(:) .* residuals.^2);
    if isnan(logL) || isinf(logL)
        logL = -1e12;
    end
catch
    logL = -1e12;
end
end

function rhat = compute_rhat(samples)
samples = squeeze(samples);
if size(samples, 1) == 1
    samples = samples.';
end
[n, m] = size(samples);
chain_means = mean(samples, 1);
B = n * var(chain_means, 1);
W = mean(var(samples, 0, 1));
var_hat = ((n - 1) / n) * W + B / n;
rhat = sqrt(var_hat / W);
end

function [ess, iact] = compute_ess_and_iact(samples)
samples = squeeze(samples);
if size(samples, 1) == 1
    samples = samples.';
end
[n, m] = size(samples);
max_lag = min(1000, n - 1);
acf_matrix = zeros(max_lag + 1, m);
for c = 1:m
    acf_matrix(:, c) = autocorr_fft(samples(:, c), max_lag);
end
rho = mean(acf_matrix(2:end, :), 2);

tau_sum = 0;
for lag = 1:max_lag
    if rho(lag) <= 0
        break;
    end
    tau_sum = tau_sum + rho(lag);
end

iact = 1 + 2 * tau_sum;
ess = (n * m) / iact;
end

function acf = autocorr_fft(x, max_lag)
x = x(:);
x = x - mean(x);
n = numel(x);
var_x = sum(x.^2) / n;
acf = zeros(max_lag + 1, 1);
if var_x <= 0
    acf(1) = 1;
    return;
end

n_fft = 2^nextpow2(2 * n);
fx = fft(x, n_fft);
acf_full = ifft(fx .* conj(fx), 'symmetric');
acf_full = acf_full(1:max_lag+1);
norm_factor = (n:-1:n-max_lag).';
acf = acf_full ./ norm_factor;
acf = acf / acf(1);
acf(1) = 1;
if any(~isfinite(acf))
    acf = zeros(max_lag + 1, 1);
    acf(1) = 1;
end
end

function write_summary_text(file_path, summary)
fid = fopen(file_path, 'w');
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Multi-chain MCMC diagnostic summary\n');
fprintf(fid, 'chains=%d\n', summary.n_chains);
fprintf(fid, 'samples_per_chain=%d\n', summary.n_samples);
fprintf(fid, 'burn_in=%d\n', summary.burn_in);
fprintf(fid, 'keep_per_chain=%d\n', summary.n_keep_per_chain);
fprintf(fid, 'valid_points=%d\n', summary.valid_points);

fprintf(fid, '\nPosterior mean\n');
fprintf(fid, 'F0_GHz=%.6f\n', summary.posterior_mean(1)/1e9);
fprintf(fid, 'BW_GHz=%.6f\n', summary.posterior_mean(2)/1e9);
fprintf(fid, 'N=%.6f\n', summary.posterior_mean(3));

fprintf(fid, '\nPosterior std\n');
fprintf(fid, 'F0_GHz=%.6f\n', summary.posterior_std(1)/1e9);
fprintf(fid, 'BW_GHz=%.6f\n', summary.posterior_std(2)/1e9);
fprintf(fid, 'N=%.6f\n', summary.posterior_std(3));

fprintf(fid, '\nPosterior 95%% interval\n');
fprintf(fid, 'F0_GHz=[%.6f, %.6f]\n', summary.posterior_ci(1,1)/1e9, summary.posterior_ci(1,2)/1e9);
fprintf(fid, 'BW_GHz=[%.6f, %.6f]\n', summary.posterior_ci(2,1)/1e9, summary.posterior_ci(2,2)/1e9);
fprintf(fid, 'N=[%.6f, %.6f]\n', summary.posterior_ci(3,1), summary.posterior_ci(3,2));

fprintf(fid, '\nPosterior MAP\n');
fprintf(fid, 'F0_GHz=%.6f\n', summary.posterior_map(1)/1e9);
fprintf(fid, 'BW_GHz=%.6f\n', summary.posterior_map(2)/1e9);
fprintf(fid, 'N=%.6f\n', summary.posterior_map(3));

fprintf(fid, '\nRhat\n');
fprintf(fid, 'F0=%.6f\n', summary.posterior_rhat(1));
fprintf(fid, 'BW=%.6f\n', summary.posterior_rhat(2));
fprintf(fid, 'N=%.6f\n', summary.posterior_rhat(3));

fprintf(fid, '\nESS\n');
fprintf(fid, 'F0=%.2f\n', summary.posterior_ess(1));
fprintf(fid, 'BW=%.2f\n', summary.posterior_ess(2));
fprintf(fid, 'N=%.2f\n', summary.posterior_ess(3));

fprintf(fid, '\nIACT\n');
fprintf(fid, 'F0=%.2f\n', summary.posterior_iact(1));
fprintf(fid, 'BW=%.2f\n', summary.posterior_iact(2));
fprintf(fid, 'N=%.2f\n', summary.posterior_iact(3));

fprintf(fid, '\nCV_percent\n');
fprintf(fid, 'F0=%.6f\n', 100 * summary.cv(1));
fprintf(fid, 'BW=%.6f\n', 100 * summary.cv(2));
fprintf(fid, 'N=%.6f\n', 100 * summary.cv(3));

fprintf(fid, '\nAcceptance_percent_by_chain\n');
for c = 1:size(summary.accept_rate, 2)
    fprintf(fid, 'chain_%d_F0=%.4f\n', c, 100 * summary.accept_rate(1, c));
    fprintf(fid, 'chain_%d_BW=%.4f\n', c, 100 * summary.accept_rate(2, c));
    fprintf(fid, 'chain_%d_N=%.4f\n', c, 100 * summary.accept_rate(3, c));
end

fprintf(fid, '\nCorrelation_matrix\n');
for i = 1:size(summary.corr_matrix, 1)
    fprintf(fid, 'row_%d=%.6f %.6f %.6f\n', i, summary.corr_matrix(i, :));
end

fprintf(fid, '\nLogL\n');
fprintf(fid, 'max=%.6f\n', summary.logL_max);
fprintf(fid, 'mean=%.6f\n', summary.logL_mean);
fprintf(fid, 'std=%.6f\n', summary.logL_std);
end

function plot_trace_and_marginals(chains, burn_in, param_names, posterior_mean, posterior_ci, figure_dir, file_name)
[n_samples, ~, n_chains] = size(chains);
colors = lines(n_chains);
figure('Color', 'w', 'Position', [80, 80, 1400, 700]);

for p = 1:3
    subplot(2, 3, p);
    hold on;
    for c = 1:n_chains
        plot(chains(:, p, c), 'Color', [colors(c, :) 0.8], 'LineWidth', 0.6);
    end
    xline(burn_in, 'k--', 'LineWidth', 1.0);
    ylabel(param_names{p});
    xlabel('Iteration');
    grid on;
    title(sprintf('Trace of %s', param_names{p}));
end

post = chains(burn_in+1:end, :, :);
combined = flatten_post_chains(post);
for p = 1:3
    subplot(2, 3, p + 3);
    histogram(combined(:, p), 50, 'Normalization', 'pdf', 'FaceColor', [0.35 0.55 0.80], 'EdgeAlpha', 0.2);
    hold on;
    xline(posterior_mean(p), 'r-', 'LineWidth', 2.0);
    xline(posterior_ci(p, 1), 'k--', 'LineWidth', 1.0);
    xline(posterior_ci(p, 2), 'k--', 'LineWidth', 1.0);
    xlabel(param_names{p});
    ylabel('PDF');
    grid on;
    title(sprintf('Posterior of %s', param_names{p}));
end

exportgraphics(gcf, fullfile(figure_dir, file_name), 'Resolution', 300);
close(gcf);
end

function plot_corner(post_chains, posterior_mean, param_names, figure_dir, file_name)
combined = flatten_post_chains(post_chains);
stride = max(floor(size(combined, 1) / 4000), 1);
combined_plot = combined(1:stride:end, :);

figure('Color', 'w', 'Position', [100, 100, 900, 850]);

subplot(3,3,1);
histogram(combined(:,1), 40, 'Normalization', 'pdf', 'FaceColor', [0.35 0.55 0.80]);
xline(posterior_mean(1), 'r--', 'LineWidth', 1.5); title(param_names{1}); ylabel('PDF'); grid on;

subplot(3,3,5);
histogram(combined(:,2), 40, 'Normalization', 'pdf', 'FaceColor', [0.35 0.70 0.45]);
xline(posterior_mean(2), 'r--', 'LineWidth', 1.5); title(param_names{2}); ylabel('PDF'); grid on;

subplot(3,3,9);
histogram(combined(:,3), 40, 'Normalization', 'pdf', 'FaceColor', [0.85 0.55 0.35]);
xline(posterior_mean(3), 'r--', 'LineWidth', 1.5); title(param_names{3}); ylabel('PDF'); xlabel(param_names{3}); grid on;

subplot(3,3,4);
scatter(combined_plot(:,1), combined_plot(:,2), 8, [0.25 0.45 0.80], 'filled', 'MarkerFaceAlpha', 0.12);
hold on; plot(posterior_mean(1), posterior_mean(2), 'r+', 'MarkerSize', 12, 'LineWidth', 2);
xlabel(param_names{1}); ylabel(param_names{2}); grid on;

subplot(3,3,7);
scatter(combined_plot(:,1), combined_plot(:,3), 8, [0.25 0.45 0.80], 'filled', 'MarkerFaceAlpha', 0.12);
hold on; plot(posterior_mean(1), posterior_mean(3), 'r+', 'MarkerSize', 12, 'LineWidth', 2);
xlabel(param_names{1}); ylabel(param_names{3}); grid on;

subplot(3,3,8);
scatter(combined_plot(:,2), combined_plot(:,3), 8, [0.25 0.45 0.80], 'filled', 'MarkerFaceAlpha', 0.12);
hold on; plot(posterior_mean(2), posterior_mean(3), 'r+', 'MarkerSize', 12, 'LineWidth', 2);
xlabel(param_names{2}); ylabel(param_names{3}); grid on;

corr_matrix = corrcoef(combined);
subplot(3,3,2); axis off; text(0.5, 0.5, sprintf('\\rho_{F0,BW}=%.3f', corr_matrix(1,2)), 'HorizontalAlignment', 'center', 'FontSize', 13);
subplot(3,3,3); axis off; text(0.5, 0.5, sprintf('\\rho_{F0,N}=%.3f', corr_matrix(1,3)), 'HorizontalAlignment', 'center', 'FontSize', 13);
subplot(3,3,6); axis off; text(0.5, 0.5, sprintf('\\rho_{BW,N}=%.3f', corr_matrix(2,3)), 'HorizontalAlignment', 'center', 'FontSize', 13);

exportgraphics(gcf, fullfile(figure_dir, file_name), 'Resolution', 300);
close(gcf);
end

function plot_posterior_reconstruction(obs, combined, posterior_mean, figure_dir, file_name)
figure('Color', 'w', 'Position', [120, 120, 900, 600]);
hold on;

f_plot = linspace(34.4e9, 37.6e9, 500);
stride = max(floor(size(combined, 1) / 100), 1);
draw_idx = 1:stride:size(combined, 1);
draw_idx = draw_idx(1:min(100, numel(draw_idx)));
for k = 1:numel(draw_idx)
    tau_draw = calculate_analytic_group_delay(f_plot, combined(draw_idx(k),1), combined(draw_idx(k),2), combined(draw_idx(k),3));
    plot(f_plot/1e9, tau_draw*1e9, 'Color', [0.90 0.70 0.70 0.20], 'LineWidth', 0.6, ...
        'HandleVisibility', 'off');
end

tau_mean = calculate_analytic_group_delay(f_plot, posterior_mean(1), posterior_mean(2), posterior_mean(3));
h_mean = plot(f_plot/1e9, tau_mean*1e9, 'r-', 'LineWidth', 2.2, 'DisplayName', 'Posterior mean reconstruction');

h_ref = [];
try
    delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
    h_ref = plot(delay_data(:,1)/1e9, delay_data(:,2)*1e9, 'k--', 'LineWidth', 1.6, 'DisplayName', 'ADS reference delay');
catch
end

h_scatter = scatter(obs.f_fit/1e9, obs.tau_fit*1e9, 44, obs.weights, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.4, ...
    'DisplayName', 'Weighted LFMCW scatter');
cb = colorbar;
ylabel(cb, 'Weight');
grid on;
xlabel('Probe frequency (GHz)');
ylabel('Group delay (ns)');
xlim([34.4, 37.6]);
ylim([0, 8]);
if isempty(h_ref)
    legend([h_mean, h_scatter], 'Location', 'northeast');
else
    legend([h_mean, h_ref, h_scatter], 'Location', 'northeast');
end

exportgraphics(gcf, fullfile(figure_dir, file_name), 'Resolution', 300);
close(gcf);
end

function combined = flatten_post_chains(post_chains)
combined = reshape(permute(post_chains, [1, 3, 2]), [], size(post_chains, 2));
end
