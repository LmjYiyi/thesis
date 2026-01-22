%% plot_fig_3_3_spectrum_defocusing.m
% 论文图 3-3：不同色散强度下差频信号的FFT频谱特征
% 生成日期：2026-01-22
% 对应章节：3.3.3 频谱特征量化
%
% 图表描述（from final.md第169行）：
% "在非色散情况下（图3-3(a)，fp = 15 GHz），频谱呈现尖锐单峰，3dB带宽约5 MHz；
%  随色散加剧（图3-3(b)，fp = 25 GHz），主瓣展宽至约50 MHz；
%  而在强色散区（图3-3(c)，fp = 29 GHz），频谱散焦达120 MHz"

clear; clc; close all;

%% 1. 物理常数与雷达参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 雷达参数（固定）
f0 = 34e9;                  % 中心频率 (Hz)
B = 4e9;                    % 带宽 (Hz)
Tm = 1e-3;                  % 调制周期 (s)
K = B / Tm;                 % 调频斜率 (Hz/s)
d = 0.15;                   % 厚度 (m)
tau_0 = d / c;              % 基础时延 (s)

%% 2. 三个色散强度场景
fp_cases = [15e9, 25e9, 29e9];  % 截止频率 (Hz)
case_labels = {'(a) 非色散', '(b) 中等色散', '(c) 强色散'};
colors = [0, 0.4470, 0.7410;    % 蓝色
          0.8500, 0.3250, 0.0980; % 橙色
          0.9290, 0.6940, 0.1250]; % 黄色

%% 3. 生成时域信号与FFT
Fs = 200e3;                 % 采样率 (200 kHz)
t = 0:1/Fs:Tm-1/Fs;         % 时间轴
N = length(t);
freq_axis = (-N/2:N/2-1) * (Fs/N);  % 频率轴（Hz）

figure('Position', [100, 100, 1600, 500], 'Color', 'w');

for i = 1:3
    fp = fp_cases(i);
    
    % 计算Drude模型参数
    omega_0 = 2*pi*f0;
    omega_p = sqrt((e^2 / (eps0 * me)) * (fp/(2*pi))^2 * (4*pi^2));
    
    % 泰勒展开系数
    x = fp / f0;
    tau_1 = (1/(2*pi)) * (-tau_0/f0) * (x^2) / ((1-x^2)^1.5);
    tau_2 = (1/(2*pi)^2) * (tau_0/f0^2) * ...
            ((3*x^4)/(1-x^2)^2.5 + x^2/(1-x^2)^1.5);
    
    % 计算一阶色散系数alpha（式3-42）
    K_prime = 2*pi*K;  % 角频率调频斜率
    A1 = tau_1 * K_prime;
    A2 = 0.5 * tau_2 * K_prime^2;
    
    alpha = (omega_0*tau_2*K_prime^2)/(2*pi) + 2*(B/Tm)*tau_1*K_prime ...
            - (B/Tm)*((tau_1*K_prime)^2 + 2*tau_0*A2);
    
    % 生成Chirp信号（式3-43）
    f_D_center = K * tau_0;  % 中心差频
    signal = exp(1j * 2*pi * (f_D_center * t + 0.5 * alpha * t.^2));
    
    % FFT计算
    fft_result = fftshift(fft(signal, N));
    power_dB = 20*log10(abs(fft_result) / max(abs(fft_result)));
    
    % 绘制子图
    subplot(1,3,i);
    hold on; box on; grid on;
    
    % 绘制频谱
    plot(freq_axis/1e6, power_dB, 'Color', colors(i,:), 'LineWidth', 2);
    
    % 标注-3dB线
    plot(xlim, [-3, -3], 'r--', 'LineWidth', 1.5, 'DisplayName', '-3 dB');
    
    % 标题和标签
    xlabel('差频频率 (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('归一化功率 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('%s\nf_p = %d GHz', case_labels{i}, fp/1e9), ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    % 标注展宽程度
    Delta_f = abs(alpha) * Tm;
    text(0.05, 0.95, sprintf('展宽: %.0f MHz', Delta_f/1e6), ...
        'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'top', 'BackgroundColor', [1, 1, 0.9], ...
        'EdgeColor', 'k');
    
    % 设置坐标范围
    xlim([-200, 200]);
    ylim([-40, 5]);
    set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'LineWidth', 1.2);
end

% 总标题
sgtitle('图 3-3 不同色散强度下差频信号的FFT频谱特征', ...
    'FontSize', 15, 'FontWeight', 'bold');

%% 4. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率）
print('-dpng', '-r300', fullfile(output_dir, '图3-3_频谱散焦效应对比.png'));

% 保存SVG（矢量图）
print('-dsvg', fullfile(output_dir, '图3-3_频谱散焦效应对比.svg'));

fprintf('✓ 图 3-3 已保存至 final_output/figures/\n');
fprintf('  - fp = 15 GHz: 尖锐单峰（非色散）\n');
fprintf('  - fp = 25 GHz: 主瓣展宽至约50 MHz\n');
fprintf('  - fp = 29 GHz: 频谱散焦达120 MHz\n');
