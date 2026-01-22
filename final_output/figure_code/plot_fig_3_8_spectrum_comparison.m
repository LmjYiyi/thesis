%% plot_fig_3_8_spectrum_comparison.m
% 论文图 3-8：工况A（安全）vs 工况B（失效）频谱对比
% 生成日期：2026-01-22
% 对应章节：3.4.2 Ka波段等离子体诊断的工程量级分析
%
% 图表描述（from final.md）：
% 工况A（fp=29 GHz, ξ=0.42）："频谱主瓣尖锐，被抑制在分辨单元内"
% 工况B（fp=32 GHz, ξ=1.275）："主瓣展宽超过127%，峰值幅度下降>10dB，
%                                 能量分散至多个FFT分辨单元，主瓣分裂或畸变漂移"
%
% 物理意义：
% - 左图：工况A的频谱（尖锐Sinc峰值）
% - 右图：工况B的频谱（展宽、衰减、分裂）

clear; clc; close all;

%% 1. 物理常数与雷达参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 雷达参数
B = 3e9;                    % 带宽 (Hz)
Tm = 1e-3;                  % 调制周期 (s)
K = B / Tm;                 % 调频斜率 (Hz/s)
fc = 35.5e9;                % 中心频率 (Hz)
d = 0.15;                   % 厚度 (m)
tau_0 = d / c;              % 基础时延 (s)

% FFT分辨率
delta_f = 1 / Tm;           % 1 kHz

%% 2. 工况A参数（安全区, fp = 29 GHz, ξ = 0.42）
fpA = 29e9;
x_A = fpA / fc;
eta_A = (B/fc) * (x_A^2) / (1-x_A^2)^1.5;
xi_A = B * eta_A * tau_0;

% 差频信号参数（工况A）
f_beat_A = K * tau_0;       % 中心差频
alpha_A = eta_A * tau_0 / Tm;  % Chirp率

%% 3. 工况B参数（失效区, fp = 32 GHz, ξ = 1.275）
fpB = 32e9;
x_B = fpB / fc;
eta_B = (B/fc) * (x_B^2) / (1-x_B^2)^1.5;
xi_B = B * eta_B * tau_0;

% 差频信号参数（工况B）
f_beat_B = K * tau_0;       % 中心差频
alpha_B = eta_B * tau_0 / Tm;  % Chirp率（更大）

%% 4. 生成时域信号与FFT
Fs = 100e3;                 % 采样率 (100 kHz)
t = 0:1/Fs:Tm-1/Fs;         % 时间轴
N = length(t);

% 工况A：差频信号（轻微Chirp）
signal_A = exp(1j * 2*pi * (f_beat_A * t + 0.5 * alpha_A * t.^2));

% 工况B：差频信号（强Chirp）
signal_B = exp(1j * 2*pi * (f_beat_B * t + 0.5 * alpha_B * t.^2));

% FFT计算
fft_A = fftshift(fft(signal_A, N));
fft_B = fftshift(fft(signal_B, N));
freq_axis = (-N/2:N/2-1) * (Fs/N);  % 频率轴（Hz）

% 归一化幅度（dB）
power_A = 20*log10(abs(fft_A) / max(abs(fft_A)));
power_B = 20*log10(abs(fft_B) / max(abs(fft_B)));

%% 5. 绘图
figure('Position', [100, 100, 1400, 600], 'Color', 'w');

% 子图1：工况A（安全区）
subplot(1,2,1);
hold on; box on; grid on;

% 绘制频谱
plot(freq_axis/1e3, power_A, 'b-', 'LineWidth', 2);

% 标注FFT分辨率
y_lim = ylim;
plot([delta_f/1e3, delta_f/1e3], [y_lim(1), -3], 'r--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('FFT分辨率 = %.1f kHz', delta_f/1e3));

% ξ 数值标注
text(0.5, -5, sprintf('判据值: ξ = %.2f < 1', xi_A), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', [0, 0.6, 0], ...
    'BackgroundColor', [0.9, 1, 0.9], 'EdgeColor', 'k');

% 标签
xlabel('差频频率 (kHz)', 'FontSize', 13);
ylabel('归一化功率 (dB)', 'FontSize', 13);
title({'(a) 工况A：安全区', sprintf('f_p = %d GHz, ξ = %.2f', fpA/1e9, xi_A)}, ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
xlim([-3, 3]);
ylim([-40, 5]);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'LineWidth', 1.2);

% 子图2：工况B（失效区）
subplot(1,2,2);
hold on; box on; grid on;

% 绘制频谱
plot(freq_axis/1e3, power_B, 'r-', 'LineWidth', 2);

% 标注FFT分辨率
y_lim = ylim;
plot([delta_f/1e3, delta_f/1e3], [y_lim(1), -3], 'g--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('FFT分辨率 = %.1f kHz', delta_f/1e3));

% 标注展宽程度
Delta_f_B = xi_B * delta_f;  % 频谱展宽
arrow_y = -10;
annotation('doublearrow', [0.58, 0.75], [0.4, 0.4], 'LineWidth', 2, 'Color', 'r');
text(1.8, arrow_y, sprintf('展宽 ≈ %.1f × δf', xi_B), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');

% ξ 数值标注
text(0.5, -5, sprintf('判据值: ξ = %.2f > 1', xi_B), ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.8, 0, 0], ...
    'BackgroundColor', [1, 0.9, 0.9], 'EdgeColor', 'k');

% 标签
xlabel('差频频率 (kHz)', 'FontSize', 13);
ylabel('归一化功率 (dB)', 'FontSize', 13);
title({'(b) 工况B：失效区', sprintf('f_p = %d GHz, ξ = %.2f', fpB/1e9, xi_B)}, ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
xlim([-3, 3]);
ylim([-40, 5]);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'LineWidth', 1.2);

% 总标题
sgtitle('图 3-8 色散效应对差频信号频谱的影响：安全区 vs 失效区对比', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% 6. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率）
print('-dpng', '-r300', fullfile(output_dir, '图3-8_频谱对比_安全vs失效.png'));

% 保存SVG（矢量图）
print('-dsvg', fullfile(output_dir, '图3-8_频谱对比_安全vs失效.svg'));

fprintf('✓ 图 3-8 已保存至 final_output/figures/\n');
fprintf('  - 工况A（安全）: ξ = %.2f < 1, 频谱尖锐\n', xi_A);
fprintf('  - 工况B（失效）: ξ = %.2f > 1, 频谱展宽 %.1f 倍\n', xi_B, xi_B);
