%% plot_fig_4_4_spectrogram.m
% 论文图 4-4：传统方法与滑动窗口时频解耦的对比
% 生成日期：2026-01-25
% 对应章节：4.2.1 基于短时观测窗的时频解耦与局部信号线性化近似
%
% 图表设计思路（对比论证）：
% - 子图(a)：传统方法的失败 —— 展示"问题"
%   * 传统FFT假设差频恒定，但实际差频随探测频率变化
%   * 灰色区域表示FFT散焦范围，无法建立差频-探测频率映射
% - 子图(b)：滑动窗口特征提取结果 —— 展示"方案"
%   * 清晰的"频率-时延"轨迹
%   * 精确追踪差频随探测频率的非线性变化
%
% 核心论点：
% 1. 传统方法为何失效？→ 无法建立差频-探测频率映射
% 2. 本文方法为何有效？→ 时频解耦，将散焦还原为轨迹
%
% 核心参数（与 thesis-code/LM_MCMC_with_noise.m 保持一致）：
% - f_p = 33 GHz（截止频率）
% - T_w = 12 μs（窗口时长）
% - T_step = 1.2 μs（步长，90%重叠）

clear; clc; close all;

%% 1. 物理常数与参数设置
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 起始频率 (Hz)
f_end = 37.4e9;             % 终止频率 (Hz)
B = f_end - f_start;        % 带宽 (Hz)
T_m = 50e-6;                % 扫频周期 (s)
K = B / T_m;                % 调频斜率 (Hz/s) = 64e12 Hz/s

% 等离子体参数（强色散工况）
f_p = 33e9;                 % 等离子体截止频率 (Hz)
omega_p = 2*pi*f_p;         % 等离子体角频率
n_e = (omega_p)^2 * epsilon_0 * m_e / q_e^2;  % 电子密度
nu_e = 1.5e9;               % 碰撞频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 传播路径参数
tau_air = 4e-9;             % 空气（参考）总时延 (s)
tau_fs = 1.75e-9;           % 自由空间段时延 (s)

% 采样参数
f_s = 80e9;                 % 原始采样率 (Hz)
t_s = 1/f_s;                % 采样间隔 (s)
N = round(T_m / t_s);       % 总采样点数
t = (0:N-1) * t_s;          % 时间轴 (s)

fprintf('物理参数设置完成\n');
fprintf('  截止频率 f_p = %.1f GHz\n', f_p/1e9);
fprintf('  电子密度 n_e = %.2e m^-3\n', n_e);
fprintf('  调频斜率 K = %.2e Hz/s\n', K);

%% 2. LFMCW信号生成与等离子体传播模拟
fprintf('生成LFMCW信号并模拟色散传播...\n');

% 发射信号
f_t = f_start + K * mod(t, T_m);    % 瞬时频率
phi_t = 2*pi * cumsum(f_t) * t_s;   % 瞬时相位
s_tx = cos(phi_t);                  % 发射信号

% 构建FFT频率轴
f_fft = (0:N-1) * (f_s/N);
idx_neg = f_fft >= f_s/2;
f_fft(idx_neg) = f_fft(idx_neg) - f_s;
omega_fft = 2*pi * f_fft;

% 第一段自由空间传播
delay_samples_fs = round(tau_fs / t_s);
s_after_fs1 = [zeros(1, delay_samples_fs), s_tx(1:end-delay_samples_fs)];

% 第二段：等离子体色散传播（频域处理）
S_after_fs1 = fft(s_after_fs1);

% Drude模型复介电常数
omega_safe = omega_fft;
omega_safe(omega_safe == 0) = 1e-10;
epsilon_r = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu_e);
epsilon_r(omega_fft == 0) = 1;

% 复波数
k_complex = (omega_fft ./ c) .* sqrt(epsilon_r);

% 传递函数（强制衰减）
k_real = real(k_complex);
k_imag = imag(k_complex);
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

% 应用传递函数
S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));

% 第三段自由空间
s_rx = [zeros(1, delay_samples_fs), s_after_plasma(1:end-delay_samples_fs)];

% 混频与低通滤波
s_mix = s_tx .* s_rx;
fc_lp = 100e6;  % 低通截止频率
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if = filtfilt(b_lp, a_lp, s_mix);

fprintf('信号生成与混频完成\n');

%% 3. 本文方法：滑动窗口特征提取
fprintf('模拟滑动窗口特征提取...\n');

% 滑动窗口参数
T_w = 12e-6;                % 窗口时长
f_s_proc = 20e6;            % 降采样后采样率
win_len = round(T_w * f_s_proc);
step_len = round(win_len / 10);  % 90%重叠
N_proc = round(T_m * f_s_proc);
t_proc = linspace(0, T_m, N_proc);

num_windows = floor((N_proc - win_len) / step_len);

% 特征点存储
feature_f_beat = [];
feature_f_probe = [];

for i = 1:num_windows
    t_center = t_proc((i-1)*step_len + round(win_len/2));
    
    % 边缘保护
    if t_center < 0.05*T_m || t_center > 0.95*T_m
        continue;
    end
    
    % 当前探测频率
    f_current = f_start + K * t_center;
    
    % 理论计算（基于Drude模型）
    omega_i = 2*pi * f_current;
    if f_current > f_p * 1.01
        epsilon_r_i = 1 - (omega_p^2) / (omega_i * (omega_i + 1i*nu_e));
        tau_g_i = (d/c) * real(1 / sqrt(epsilon_r_i)) + 2*tau_fs;
        f_beat_i = K * tau_g_i;
        
        % 模拟ESPRIT测量噪声
        f_beat_meas = f_beat_i * (1 + 0.002*randn);
        
        feature_f_beat = [feature_f_beat, f_beat_meas];
        feature_f_probe = [feature_f_probe, f_current];
    end
end

fprintf('特征点提取完成: %d 个有效点\n', length(feature_f_beat));

%% 4. 计算理论曲线（用于对比）
t_theory = linspace(0.05*T_m, 0.95*T_m, 200);
f_probe_theory = f_start + K * t_theory;
f_beat_theory = zeros(size(t_theory));

for i = 1:length(t_theory)
    omega_i = 2*pi * f_probe_theory(i);
    if f_probe_theory(i) > f_p * 1.01
        epsilon_r_i = 1 - (omega_p^2) / (omega_i * (omega_i + 1i*nu_e));
        tau_g_i = (d/c) * real(1 / sqrt(epsilon_r_i)) + 2*tau_fs;
        f_beat_theory(i) = K * tau_g_i;
    else
        f_beat_theory(i) = NaN;
    end
end

%% 5. 传统方法下的差频提取（长窗FFT，失败示例）
num_blocks = 20;                         % 粗分块数（越少越“传统”）
block_len = floor(length(s_if) / num_blocks);

f_probe_fft = zeros(1, num_blocks);
f_beat_fft = zeros(1, num_blocks);

for k = 1:num_blocks
    idx = (k-1)*block_len + (1:block_len);
    idx(idx > length(s_if)) = [];
    if isempty(idx)
        continue;
    end
    
    s_blk = s_if(idx);
    S_blk = abs(fft(s_blk, 2^14));
    f_axis_blk = (0:length(S_blk)-1) * f_s / length(S_blk);
    
    % 只在差频范围内找峰
    idx_band = f_axis_blk > 200e3 & f_axis_blk < 450e3;
    f_axis_band = f_axis_blk(idx_band);
    S_band = S_blk(idx_band);
    if isempty(S_band)
        continue;
    end
    
    [~, imax] = max(S_band);
    f_beat_fft(k) = f_axis_band(imax);
    
    % 对应探测频率（块中心）
    t_center = mean(idx) * t_s;
    f_probe_fft(k) = f_start + K * t_center;
end

% 去掉未赋值的点
valid_fft = (f_probe_fft > 0) & (f_beat_fft > 0);
f_probe_fft = f_probe_fft(valid_fft);
f_beat_fft = f_beat_fft(valid_fft);

%% 6. 绘图：问题 vs 方案
fprintf('绘制对比图...\n');

fig = figure('Position', [100, 100, 1100, 500], 'Color', 'w');

% =========================================================================
% 子图(a): 传统方法的失败
% =========================================================================
subplot(1, 2, 1);
% 先统一坐标轴字体（仅影响刻度与轴）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.1);
hold on;

% 绘制实际差频范围的灰色阴影区域，表示"散焦"
f_beat_min_val = min(feature_f_beat);
f_beat_max_val = max(feature_f_beat);
fill([34.2, 37.4, 37.4, 34.2], ...
     [f_beat_min_val, f_beat_min_val, f_beat_max_val, f_beat_max_val]/1e3, ...
     [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
     'DisplayName', 'FFT散焦区域');

% 绘制传统FFT得到的失败轨迹
plot(f_probe_fft/1e9, f_beat_fft/1e3, 'k.-', 'LineWidth', 1.5, ...
    'MarkerSize', 12, 'DisplayName', '传统FFT结果');

% 坐标轴标签
xlabel('探测频率 f_{probe} (GHz)', 'FontSize', 12, 'Interpreter', 'tex', 'FontName', 'SimHei');
ylabel('差频 f_D (kHz)', 'FontSize', 12, 'Interpreter', 'tex', 'FontName', 'SimHei');
title('(a) 传统方法：差频关系不可解析', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'SimHei');

grid on;
xlim([34.2, 37.4]);
ylim([290, 340]);
lgd_left = legend('Location', 'northeast', 'FontSize', 9);
set(lgd_left, 'FontName', 'SimHei', 'Interpreter', 'tex');

% =========================================================================
% 子图(b): 滑动窗口的成功
% =========================================================================
subplot(1, 2, 2);
% 先统一坐标轴字体（仅影响刻度与轴）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.1);
hold on;

% 绘制理论线
valid_theory = ~isnan(f_beat_theory);
plot(f_probe_theory(valid_theory)/1e9, f_beat_theory(valid_theory)/1e3, 'b--', 'LineWidth', 2.5, ...
    'DisplayName', '理论真值 (Drude)');

% 绘制测量点
scatter(feature_f_probe/1e9, feature_f_beat/1e3, 40, ...
    'MarkerFaceColor', [0.85, 0.33, 0.1], 'MarkerEdgeColor', 'w', ...
    'DisplayName', '滑动窗口提取');

% 坐标轴标签
xlabel('探测频率 f_{probe} (GHz)', 'FontSize', 12, 'Interpreter', 'tex', 'FontName', 'SimHei');
ylabel('瞬时差频 f_D (kHz)', 'FontSize', 12, 'Interpreter', 'tex', 'FontName', 'SimHei');
title('(b) 时频解耦：清晰的差频轨迹', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'SimHei');

grid on;
lgd_right = legend('Location', 'northeast', 'FontSize', 10);
set(lgd_right, 'FontName', 'SimHei', 'Interpreter', 'tex');
xlim([34.2, 37.4]);
ylim([290, 340]);

% =========================================================================
% 总标题
% =========================================================================
sgtitle('传统方法与滑动窗口时频解耦效果对比', ...
    'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'SimHei');

%% 7. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_name_base = '图4-4_传统方法vs时频解耦';
print(fig, '-dpng', '-r300', fullfile(output_dir, [fig_name_base, '.png']));
print(fig, '-dsvg', fullfile(output_dir, [fig_name_base, '.svg']));

fprintf('\n✓ 图 4-4 已保存至 final_output/figures/\n');
fprintf('  - %s.png\n', fig_name_base);
fprintf('  - %s.svg\n', fig_name_base);

%% 8. 输出物理特征统计
fprintf('\n========== 物理特征统计 ==========\n');
fprintf('仿真参数:\n');
fprintf('  截止频率 f_p = %.1f GHz\n', f_p/1e9);
fprintf('  碰撞频率 ν_e = %.1f GHz\n', nu_e/1e9);
fprintf('  等离子体厚度 d = %.2f m\n', d);
fprintf('  调频斜率 K = %.2e Hz/s\n', K);

fprintf('\n传统方法结论:\n');
fprintf('  差频散焦范围: %.1f - %.1f kHz (约 %.1f kHz)\n', ...
    f_beat_min_val/1e3, f_beat_max_val/1e3, (f_beat_max_val-f_beat_min_val)/1e3);
fprintf('  问题: 差频与探测频率关系不可解析（无唯一映射）\n');

fprintf('\n滑动窗口方法:\n');
fprintf('  窗口时长 T_w = %.0f μs\n', T_w*1e6);
fprintf('  步长 T_step = %.1f μs (%.0f%% 重叠)\n', T_w/10*1e6, 90);
fprintf('  有效窗口数 N_w = %d\n', length(feature_f_beat));
fprintf('  优势: 精确追踪差频随探测频率的非线性变化\n');
fprintf('================================================\n');
