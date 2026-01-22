%% plot_fig_3_2_time_frequency_spectrogram.m
% 论文图 3-2：差频信号瞬时频率演化的时频图（Spectrogram）
% 生成日期：2026-01-22
% 对应章节：3.3.2 差频信号相位的非线性畸变
%
% 图表建议（from 用户上传图片）：
% "绘制Spectrogram来展示 fD(t)=fD'+αt 这一动态过程"
% "图表作用：直观展示理想信号是一条水平线，而色散信号是一条斜线（Chirp），对比即为α"

clear; clc; close all;

%% 1. 物理常数与雷达参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 雷达参数
f0 = 34e9;                  % 中心频率 (Hz)
B = 4e9;                    % 带宽 (Hz)
Tm = 1e-3;                  % 调制周期 (s)
K = B / Tm;                 % 调频斜率 (Hz/s)
d = 0.15;                   % 厚度 (m)
tau_0 = d / c;              % 基础时延 (s)

%% 2. 两种场景对比
% 场景1：无色散（α=0）
% 场景2：强色散（fp=29 GHz, α≠0）

fp_strong = 29e9;           % 强色散截止频率

% 计算强色散参数
omega_0 = 2*pi*f0;
x = fp_strong / f0;
tau_1 = (1/(2*pi)) * (-tau_0/f0) * (x^2) / ((1-x^2)^1.5);
tau_2 = (1/(2*pi)^2) * (tau_0/f0^2) * ...
        ((3*x^4)/(1-x^2)^2.5 + x^2/(1-x^2)^1.5);

K_prime = 2*pi*K;
A1 = tau_1 * K_prime;
A2 = 0.5 * tau_2 * K_prime^2;

alpha_strong = (omega_0*tau_2*K_prime^2)/(2*pi) + 2*(B/Tm)*tau_1*K_prime ...
               - (B/Tm)*((tau_1*K_prime)^2 + 2*tau_0*A2);

fprintf('计算参数：\n');
fprintf('  无色散：α = 0 Hz/s\n');
fprintf('  强色散：α = %.2e Hz/s\n', alpha_strong);
fprintf('  频率漂移速率：%.2f MHz/ms\n', abs(alpha_strong)*Tm/1e6);

%% 3. 生成时域信号
Fs = 200e3;                 % 采样率 (200 kHz)
t = 0:1/Fs:Tm-1/Fs;         % 时间轴
N = length(t);

% 中心差频
f_beat_center = K * tau_0;

% 无色散信号（水平线）
signal_nodispersion = exp(1j * 2*pi * f_beat_center * t);

% 强色散信号（Chirp，斜线）
signal_dispersion = exp(1j * 2*pi * (f_beat_center * t + 0.5 * alpha_strong * t.^2));

%% 4. 时频分析（Spectrogram）
window_len = 256;           % 窗口长度
overlap = round(0.95 * window_len);  % 重叠95%（平滑）
nfft = 512;

% 计算时频图
[S_nodis, F_nodis, T_nodis] = spectrogram(signal_nodispersion, ...
    hann(window_len), overlap, nfft, Fs, 'yaxis');

[S_dis, F_dis, T_dis] = spectrogram(signal_dispersion, ...
    hann(window_len), overlap, nfft, Fs, 'yaxis');

%% 5. 绘图
figure('Position', [100, 100, 1400, 600], 'Color', 'w');

% 子图1：无色散（水平线）
subplot(1,2,1);
imagesc(T_nodis*1e3, F_nodis/1e3, 20*log10(abs(S_nodis)));
axis xy;
colormap(jet);
caxis([-60, 0]);
colorbar;

hold on;
% 叠加理论瞬时频率线
plot(t*1e3, ones(size(t))*f_beat_center/1e3, 'w--', 'LineWidth', 2.5, ...
    'DisplayName', '理论: f_D(t) = f_D''');
hold off;

xlabel('时间 (ms)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('频率 (kHz)', 'FontSize', 13, 'FontWeight', 'bold');
title({'(a) 无色散：瞬时频率恒定', 'α = 0 (水平线)'}, ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11, 'TextColor', 'w');
set(gca, 'FontName', 'SimHei', 'FontSize', 12, 'LineWidth', 1.2);
grid on;

% 子图2：强色散（斜线Chirp）
subplot(1,2,2);
imagesc(T_dis*1e3, F_dis/1e3, 20*log10(abs(S_dis)));
axis xy;
colormap(jet);
caxis([-60, 0]);
colorbar;

hold on;
% 叠加理论瞬时频率线（线性漂移）
f_inst = f_beat_center + alpha_strong * t;
plot(t*1e3, f_inst/1e3, 'w--', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('理论: f_D(t) = f_D'' + αt\n(斜率 α = %.1e Hz/s)', alpha_strong));
hold off;

% 标注斜率α
arrow_x = [0.3, 0.7];
arrow_y = [f_beat_center/1e3 + 20, f_beat_center/1e3 + 80];
annotation('textarrow', [0.65, 0.75], [0.5, 0.65], ...
    'String', sprintf('斜率 α\n频率漂移'), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w', 'LineWidth', 2);

xlabel('时间 (ms)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('频率 (kHz)', 'FontSize', 13, 'FontWeight', 'bold');
title({'(b) 强色散：瞬时频率线性漂移', sprintf('f_p = %d GHz, α ≠ 0 (斜线Chirp)', fp_strong/1e9)}, ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11, 'TextColor', 'w', 'Box', 'off');
set(gca, 'FontName', 'SimHei', 'FontSize', 12, 'LineWidth', 1.2);
grid on;

% 总标题
sgtitle('图 3-2 差频信号瞬时频率演化的时频分析对比', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% 6. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率）
print('-dpng', '-r300', fullfile(output_dir, '图3-2_时频分析对比.png'));

% 保存SVG（矢量图）
print('-dsvg', fullfile(output_dir, '图3-2_时频分析对比.svg'));

fprintf('\n✓ 图 3-2 已保存至 final_output/figures/\n');
fprintf('  - 左图：无色散，瞬时频率为水平线（α=0）\n');
fprintf('  - 右图：强色散，瞬时频率呈斜线Chirp（α≠0）\n');
fprintf('  - 对比即为α的可视化\n');
