% compare_s21_groupdelay_vs_lfm.m
% 读取 .s2p (MA 格式)，计算 S21 群时延
% 用简化 LFMCW 仿真（IF 级别）提取时延点并与群时延作图比较
% 适配你的 LM_MCMC.m 参数（f_start, f_end, T）

clear; close all; clc;

%% ========== 1. 参数（与 LM_MCMC.m 对齐） ==========
s2p_filename = 'ka_bandpassfilter.s2p'; % 假设当前目录有这份文件（根据你示例）
% LFMCW 参数（来自 LM_MCMC.m 片段）
f_start = 34.2e9;        % Hz
f_end   = 37.4e9;        % Hz
T_chirp = 50e-6;         % s (单次上行/下行扫频时长)
B = f_end - f_start;     % 带宽
K = B / T_chirp;         % 斜率 (Hz/s)

%% ========== 2. 读取 .s2p（MA 格式）并解析 S21 ==========
if ~exist(s2p_filename,'file')
    error('找不到 %s。请把你的 .s2p 放到当前目录或修改脚本中的文件名。', s2p_filename);
end

% 打开并解析
fid = fopen(s2p_filename,'r');
freq_GHz = [];
s21_mag = [];
s21_ang_deg = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line)
        continue;
    end
    if startsWith(line,'!') || startsWith(line,'! ')
        % 注释行跳过
        continue;
    end
    if startsWith(line,'#')
        % header, 可解析单位、格式（本脚本假设 # GHz S MA ...）
        header = line;
        continue;
    end
    % 数据行：按你给的示例，每行 9 列（freq, S11_mag, S11_ang, S21_mag, S21_ang, S12_mag, S12_ang, S22_mag, S22_ang）
    cols = textscan(line, '%f');
    vals = cols{1};
    if numel(vals) >= 9
        freq_GHz(end+1,1) = vals(1);
        s21_mag(end+1,1)  = vals(4);
        s21_ang_deg(end+1,1)= vals(5);
    else
        % 如果空格分隔不规则，尝试更强的解析
        toks = strsplit(line);
        if numel(toks) >= 9
            freq_GHz(end+1,1) = str2double(toks{1});
            s21_mag(end+1,1)  = str2double(toks{4});
            s21_ang_deg(end+1,1)= str2double(toks{5});
        else
            % 忽略非数据行
            continue;
        end
    end
end
fclose(fid);

% 单位转换，频率转为 Hz
freq = freq_GHz * 1e9;

% S21 复数表示（MA: magnitude & angle in degrees）
S21 = s21_mag .* exp(1j * deg2rad(s21_ang_deg));

%% ========== 3. 计算相位、unwrap、群时延 ==========
phi = angle(S21);           % rad (在 -pi..pi)
phi_unwrap = unwrap(phi);   % 连续相位 (rad)

% 为了更稳健的数值求导，可对相位做小平滑（可选）
% phi_unwrap_smooth = sgolayfilt(phi_unwrap,3,11); % 仅在数据点足够多时使用
phi_unwrap_smooth = phi_unwrap;

% dphi/df (频率为 Hz)
dphi_df = gradient(phi_unwrap_smooth) ./ gradient(freq);

% 群时延： tau_g = - (1/(2*pi)) * dphi/df  (单位 s)
tau_g = - dphi_df ./ (2*pi);

% 可视化（先准备）
% 将群时延单位转为 ns 便于观察
tau_g_ns = tau_g * 1e9;

%% ========== 4. 简化 LFMCW 仿真（IF 级别）并提取时延 ==========
% 我们以 "点目标" 模拟：给定若干真实时延 tau_true，生成 IF tone(s)
% IF tone frequency f_b = K * tau_true
% 现实中若存在多个目标，IF 为多音或簇；我们用 FFT 找峰并估计 f_b

% 设定若干目标延迟（示例：取 LM_MCMC.m 中的 tau_air）
tau_true = [4e-9, 8e-9];  % s, 例子：4 ns、8 ns 的两个回波
amp_true = [1.0, 0.4];     % 回波幅度比例（可调）

% IF 采样参数（IF 信号仿真） - 只需满足对 beat 频率 f_b 的采样
% 最大可能的 f_b = K * max(tau_true). 选择 Fs_if 足够大以解析峰值
f_b_max = K * max(tau_true);
Fs_if = max(5 * f_b_max, 1e6);   % sps (至少 > 5*max beat), 最小 1 MHz
% 为了高分辨率 FFT，选择较长的采样时间等于一条 chirp
t = 0 : 1/Fs_if : T_chirp - 1/Fs_if;  % 单次上行 chirp 的采样时间轴

% 生成 IF 信号：IF(t) = sum_k a_k * exp(1j*2*pi*f_b_k * t)  (可以加噪声)
s_if = zeros(size(t));
for k = 1:length(tau_true)
    f_b_k = K * tau_true(k);
    s_if = s_if + amp_true(k) * exp(1j*2*pi*f_b_k .* t);
end

% 添加少量噪声（可选）
SNR_dB = 30;
noise_pow = 10^(-SNR_dB/10);
s_if = s_if + sqrt(noise_pow/2) * (randn(size(s_if)) + 1j*randn(size(s_if)));

% 为了提高 FFT 频率分辨率，做零填充
Nfft = 2^20; % 1M 点左右（根据内存可调整）
S_IF = fft(s_if, Nfft);
f_axis = (0:Nfft-1) * (Fs_if / Nfft); % Hz

% 只看正频率到 Nyquist
half = 1:floor(Nfft/2);
mag_spec = abs(S_IF(half));
f_axis_pos = f_axis(half);

% 找出若干个最大峰（对应不同目标）
% 简单 peak-picking：找局部极大并排序
[~, idx_sorted] = sort(mag_spec,'descend');
n_peaks = length(tau_true); % 我们希望找这么多个峰
peak_indices = [];
cnt = 1;
i_try = 1;
while (cnt <= n_peaks) && (i_try <= length(idx_sorted))
    idx = idx_sorted(i_try);
    % 防止重复近邻峰：要求与已有峰间隔至少 5 bins
    if all(abs(idx - peak_indices) > 5)
        peak_indices(end+1) = idx; %#ok<SAGROW>
        cnt = cnt + 1;
    end
    i_try = i_try + 1;
end

peak_freqs = f_axis_pos(peak_indices);    % Hz (这些是 IF 的峰值频率)
% 若要更精确 f_peak，可用相位插值或 centroid 局部拟合；这里用二次插值提高精度
peak_freqs_refined = zeros(size(peak_freqs));
for i = 1:length(peak_indices)
    k = peak_indices(i);
    if k>1 && k<length(mag_spec)
        % 3 点二次插值（parabolic interpolation）
        alpha = mag_spec(k-1);
        beta  = mag_spec(k);
        gamma = mag_spec(k+1);
        p = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma); % offset in bins
        peak_freqs_refined(i) = f_axis_pos(k) + p * (Fs_if/Nfft);
    else
        peak_freqs_refined(i) = f_axis_pos(k);
    end
end

% 对应时延估计
tau_est = peak_freqs_refined / K;  % s

% 为了与 S21 频率轴对应，我们需要定义每个时延点对应的"探测频率" f_probe
% 一个简单且常用的做法：取接收时刻为扫频中点 t_center = T_chirp/2
% 此时发射瞬时频率约为 f_tx = f_start + K * (t_center - tau)
t_center = T_chirp/2;
f_probe_est = f_start + K * (t_center - tau_est);

% 将结果按频率排序以便绘图
[f_probe_est_sorted, si] = sort(f_probe_est);
tau_est_sorted = tau_est(si);

%% ========== 5. 将 S21 的群时延和 LFMCW 提取点绘成一张图 ==========
% 对 S21 群时延做平滑（可选）
tau_g_ns_smooth = movmedian(tau_g_ns, 11);

figure('Color','w','Position',[100 100 900 500]);
hold on;
plot(freq/1e9, tau_g_ns_smooth, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Group delay from S21 (ns)');
scatter(f_probe_est_sorted/1e9, tau_est_sorted*1e9, 80, 'b', 'filled', 'DisplayName', 'LFMCW extracted delay (ns)');
xlabel('Frequency (GHz)');
ylabel('Delay (ns)');
title('Compare S21 group delay vs LFMCW extracted delay points');
legend('Location','best');
grid on;
xlim([min(freq)/1e9, max(freq)/1e9]);
hold off;

% 保存图像
saveas(gcf,'groupdelay_vs_lfm_compare.png');

%% ========== 6. 输出数值表格（并打印到 Command Window） ==========
T = table((freq/1e9), (tau_g_ns), 'VariableNames', {'Freq_GHz','GroupDelay_ns'});
fprintf('\n--- S21 group delay sample (first 10 rows) ---\n');
disp(T(1:min(10,height(T)),:));

fprintf('\n--- LFMCW extracted points ---\n');
T2 = table(f_probe_est_sorted'/1e9, tau_est_sorted'*1e9, 'VariableNames', {'Freq_GHz','Tau_ns'});
disp(T2);

% 将 LFMCW 点写入 csv 以便进一步处理/论文制图
writetable(T2, 'lfmcw_extracted_delays.csv');

fprintf('\n已保存图像: groupdelay_vs_lfm_compare.png\n已保存 LFMCW 点: lfmcw_extracted_delays.csv\n\n');

%% ========== 结束语 ==========
% 说明：
% - 若想用实际 LM_MCMC.m 中更精细的 RF/Drude/传递函数处理替换 IF 简化模型：
%   可把 LM_MCMC.m 中生成的 s_tx（在 RF 或基带）通过 S21(H) 插值乘法（频域），ifft 得到 s_rx，
%   再与 s_tx 混频得到 s_if，然后重复上面的 FFT 峰值提取流程。
% - 若需要我把这段“更精确的 RF->频域乘以 S21->ifft->混频”部分直接添加进来，请回复“加精确传输仿真”。
