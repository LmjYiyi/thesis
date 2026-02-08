%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW + S21 滤波器：读取介质 S21，仿真 LFMCW 经滤波器传播，
% 用 LM_MCMC 流程做特征提取（滑动窗 + MDL + ESPRIT），
% 将提取的时延点与 S21 群时延曲线对比（不做参数反演）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% ========== 1. 参数（与 LM_MCMC.m 对齐） ==========
s2p_filename = 'ka_bandpassfilter.s2p';

% LFMCW 雷达参数（与 LM_MCMC 一致；T_m 略短以加速本脚本）
f_start = 34.2e9;             % Hz
f_end   = 37.4e9;             % Hz
T_m     = 20e-6;              % s（缩短以加速，仍足够提取时延-频率关系）
B       = f_end - f_start;
K       = B / T_m;
f_s     = 80e9;               % 采样率 Hz

t_s     = 1 / f_s;
N       = round(T_m / t_s);
t       = (0 : N-1) * t_s;

% FFT 频率轴（含负频率，与 LM_MCMC 一致）
f_fft   = (0 : N-1) * (f_s / N);
idx_neg = f_fft >= f_s/2;
f_fft(idx_neg) = f_fft(idx_neg) - f_s;
omega_fft = 2*pi*f_fft;

%% ========== 2. 读取 .s2p（MA 格式）并解析 S21 ==========
if ~exist(s2p_filename, 'file')
    error('找不到 %s。请将 .s2p 放在当前目录或修改 s2p_filename。', s2p_filename);
end

fid = fopen(s2p_filename, 'r');
freq_GHz = [];
s21_mag  = [];
s21_ang_deg = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line), continue; end
    if startsWith(line, '!') || startsWith(line, '#')
        continue;
    end
    cols = textscan(line, '%f');
    vals = cols{1};
    if numel(vals) >= 5
        freq_GHz(end+1, 1) = vals(1);
        s21_mag(end+1, 1)  = vals(4);
        s21_ang_deg(end+1, 1) = vals(5);
    end
end
fclose(fid);

freq_s2p = freq_GHz * 1e9;   % Hz
S21_complex = s21_mag .* exp(1j * deg2rad(s21_ang_deg));

fprintf('S2P 读取完成: %d 点, 频率范围 %.3f–%.3f GHz\n', ...
    length(freq_s2p), min(freq_s2p)/1e9, max(freq_s2p)/1e9);

%% ========== 3. 由 S21 计算群时延 (回归稳健版) ==========
phi_s21 = angle(S21_complex);
phi_unwrap = unwrap(phi_s21);

% 1. 先对相位做一点点预平滑 (去除极微小的量化噪声)
phi_unwrap = smoothdata(phi_unwrap, 'rloess', 5);

% 2. 使用 gradient 计算导数 (它能自动处理非均匀频率间隔)
dphi = gradient(phi_unwrap);
df   = gradient(freq_s2p);
tau_g_s21 = - (dphi ./ df) / (2*pi); % s

% 3. 后处理平滑 (关键步骤)
% 根据数据点密度自适应窗口：约占总点数的 2%
span = round(length(freq_s2p) * 0.02);
if span < 5, span = 5; end
% 使用 'gaussian' 或 'rloess' 能够很好地保留波峰形状同时滤除毛刺
tau_g_ns = smoothdata(tau_g_s21 * 1e9, 'gaussian', span);

fprintf('群时延计算完成 (Gradient + Gaussian Smooth)\n');

%% ========== 4. LFMCW 发射信号生成（与 LM_MCMC 一致） ==========
f_t = f_start + K * mod(t, T_m);
phi_t = 2*pi * cumsum(f_t) * t_s;
s_tx = cos(phi_t);
S_tx = fft(s_tx);

%% ========== 5. 用 S21 作为信道 H(f)：向量化插值到 FFT 频率轴 ==========
% S2P 仅含正频率；实系统 H(-f)=conj(H(f))（避免百万次循环）
H_fft = zeros(size(f_fft));
f_s2p_min = min(freq_s2p);
f_s2p_max = max(freq_s2p);

pos_idx = (f_fft >= f_s2p_min) & (f_fft <= f_s2p_max);
neg_idx = (-f_fft >= f_s2p_min) & (-f_fft <= f_s2p_max);
H_fft(pos_idx) = interp1(freq_s2p, S21_complex, f_fft(pos_idx), 'linear', 'extrap');
H_fft(neg_idx) = conj(interp1(freq_s2p, S21_complex, -f_fft(neg_idx), 'linear', 'extrap'));

% 应用信道
S_rx = S_tx .* H_fft;
s_rx = real(ifft(S_rx));

%% ========== 6. 混频与低通（与 LM_MCMC 一致） ==========
s_mix = s_tx .* s_rx;
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if = filtfilt(b_lp, a_lp, s_mix);

%% ========== 7. 特征提取：滑动窗 + MDL + TLS-ESPRIT（与 LM_MCMC 一致） ==========
decimation_factor = 200;
f_s_proc = f_s / decimation_factor;   % 400 MHz
s_proc = s_if(1 : decimation_factor : end);
t_proc = t(1 : decimation_factor : end);
N_proc = length(s_proc);

% 【核心修改】寻找平衡点
% 太小(0.8us) -> 容不下长波(低频差拍)，算不出时延
% 太大(8.0us) -> 频率分辨率低，双峰会被抹平
% 折中值: 2.4us。对于 5ns 时延 (800kHz差拍)，能包含约 2 个周期，ESPRIT 可解。
win_time = 2.4e-6;

win_len = round(win_time * f_s_proc);
step_len = round(win_len / 4);
L_sub = round(win_len / 2);

feature_f_probe = [];
feature_tau_absolute = [];
feature_amplitude = [];

num_windows = floor((N_proc - win_len) / step_len) + 1;
hWait = waitbar(0, 'ESPRIT 特征提取中...');

for i = 1 : num_windows
    idx_start = (i-1)*step_len + 1;
    idx_end = idx_start + win_len - 1;
    if idx_end > N_proc, break; end

    x_window = s_proc(idx_start : idx_end);
    t_center = t_proc(idx_start + round(win_len/2));
    f_current_probe = f_start + K * t_center;

    if t_center > 0.95*T_m || t_center < 0.05*T_m, continue; end

    M_sub = win_len - L_sub + 1;
    X_hankel = zeros(L_sub, M_sub);
    for k = 1 : M_sub
        X_hankel(:, k) = x_window(k : k+L_sub-1).';
    end

    R_f = (X_hankel * X_hankel') / M_sub;
    J_mat = fliplr(eye(L_sub));
    R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;

    [eig_vecs, eig_vals_mat] = eig(R_x);
    lambda = diag(eig_vals_mat);
    [lambda, sort_idx] = sort(lambda, 'descend');
    eig_vecs = eig_vecs(:, sort_idx);

    % MDL
    p = length(lambda);
    N_snaps = M_sub;
    mdl_cost = zeros(p, 1);
    for k = 0 : p-1
        noise_evals = lambda(k+1:end);
        noise_evals(noise_evals < 1e-15) = 1e-15;
        g_mean = prod(noise_evals)^(1/length(noise_evals));
        a_mean = mean(noise_evals);
        term1 = -(p-k)*N_snaps*log(g_mean/a_mean);
        term2 = 0.5*k*(2*p-k)*log(N_snaps);
        mdl_cost(k+1) = term1 + term2;
    end
    [~, min_idx] = min(mdl_cost);
    k_est = min_idx - 1;
    num_sources = max(1, min(k_est, 3));

    % TLS-ESPRIT
    Us = eig_vecs(:, 1:num_sources);
    psi = (Us(1:end-1,:)' * Us(1:end-1,:)) \ (Us(1:end-1,:)' * Us(2:end,:));
    z_roots = eig(psi);
    est_freqs = abs(angle(z_roots) * f_s_proc / (2*pi));

    % 稍微放宽一点低频掩码，防止把真实的低频点滤掉
    valid_mask = (est_freqs > 20e3) & (est_freqs < 20e6);
    valid_freqs = est_freqs(valid_mask);
    if isempty(valid_freqs), continue; end

    [f_beat_est, ~] = min(valid_freqs);
    amp_est = rms(x_window);
    tau_est = f_beat_est / K;

    feature_f_probe   = [feature_f_probe,   f_current_probe];
    feature_tau_absolute = [feature_tau_absolute, tau_est];
    feature_amplitude = [feature_amplitude, amp_est];

    if mod(i, 10) == 0 || i == num_windows
        waitbar(i/num_windows, hWait);
    end
end
if ishandle(hWait), close(hWait); end

fprintf('特征提取完成: %d 个 (f_probe, tau) 点\n', length(feature_f_probe));

%% ========== 8. 绘图：S21 群时延曲线 vs LFMCW 提取的时延点 ==========
% 中英混排：坐标轴用 Times New Roman，中文用 SimHei
figure('Color', 'w', 'Position', [100 100 900 520]);

% 先设坐标轴字体
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
hold on;

% 1. 绘制 S21 群时延曲线
tau_g_ns_plot = movmedian(tau_g_ns, 11);
plot(freq_s2p/1e9, tau_g_ns_plot, 'r-', 'LineWidth', 2, 'DisplayName', 'S21 群时延');

% 2. 绘制 LFMCW 提取的时延点
valid_delay = feature_tau_absolute > 0 & feature_tau_absolute < 500e-9;
scatter(feature_f_probe(valid_delay)/1e9, feature_tau_absolute(valid_delay)*1e9, 24, 'b', 'filled', ...
    'DisplayName', 'LFMCW 特征时延');

xlabel('频率 (GHz)', 'FontName', 'SimHei', 'FontSize', 12);
ylabel('时延 (ns)', 'FontName', 'SimHei', 'FontSize', 12);
title('S21 群时延与 LFMCW 提取时延对比（滤波器介质）', 'FontName', 'SimHei', 'FontSize', 14);
legend('Location', 'best', 'FontName', 'SimHei', 'FontSize', 11);
grid on;
xlim([min(freq_s2p)/1e9, max(freq_s2p)/1e9]);

% ====== 【核心修正】分步处理，确保 100% 是列向量再拼接 ======
% 第一步：确保 S21 数据是列向量
vec1 = tau_g_ns(:);

% 第二步：提取有效点 -> 确保转为列向量 -> 乘以单位
vals_extracted = feature_tau_absolute(valid_delay);
vec2 = vals_extracted(:) * 1e9;

% 第三步：拼接（现在两个都是 n x 1，绝对安全）
tau_plot_vals = [vec1; vec2];
% ========================================================

% 过滤无穷大值并设置 Y 轴范围
tau_plot_vals = tau_plot_vals(isfinite(tau_plot_vals));
if ~isempty(tau_plot_vals)
    ylim([max(0, min(tau_plot_vals) - 2), max(tau_plot_vals) + 5]);
end
hold off;

% 保存
saveas(gcf, 'groupdelay_vs_lfmcw_extracted.png');
saveas(gcf, 'groupdelay_vs_lfmcw_extracted.fig');
fprintf('已保存: groupdelay_vs_lfmcw_extracted.png / .fig\n');

%% ========== 9. 可选：将提取点与 S21 群时延插值对比（数值） ==========
if ~isempty(feature_f_probe) && sum(valid_delay) > 0
    f_valid = feature_f_probe(valid_delay);
    tau_valid = feature_tau_absolute(valid_delay) * 1e9;

    % 修正：将 'nan' 改为 NaN (数值常量)
    % 语法: interp1(x, v, xq, method, extrapolation_value)
    tau_s21_at_probe = interp1(freq_s2p/1e9, tau_g_ns, f_valid/1e9, 'linear', NaN);

    err_ns = tau_valid - tau_s21_at_probe;

    fprintf('LFMCW 时延 vs S21 群时延: 平均偏差 %.3f ns, 标准差 %.3f ns\n', ...
        mean(err_ns, 'omitnan'), std(err_ns, 'omitnan'));
end
