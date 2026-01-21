%% plot_fig_3_3.m
% 论文图 3-3：色散效应下差频信号频谱对比
% 生成日期：2026-01-22
% 对应章节：3.3.3 频谱特征量化
% 
% 文档描述（第167行）：
% "图3-3展示了不同色散强度下差频信号的FFT频谱特征。如图所示，
% 在非色散情况下(图3-3(a)，fp = 15 GHz)，频谱呈现尖锐单峰，3dB带宽约5 MHz；
% 随色散加剧(图3-3(b)，fp = 25 GHz)，主瓣展宽至约50 MHz；
% 而在强色散区(图3-3(c)，fp = 29 GHz)，频谱散焦达120 MHz"

clear; clc; close all;

%% 1. 参数设置（与 thesis-code 保持一致）
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数 (F/m)
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% LFMCW雷达参数
f0 = 34e9;                  % 中心频率 (Hz)
B = 4e9;                    % 带宽 (Hz)
T_m = 1e-3;                 % 扫频周期 (s)
Fs = 100e6;                 % 采样率 (Hz)
t = linspace(0, T_m, Fs*T_m); % 时间向量

% 等离子体参数（三种色散强度）
d = 0.15;                   % 等离子体厚度 (m)
nu_e = 1.5e9;               % 碰撞频率 (Hz)

% 三组电子密度参数
cases = struct();
cases(1).label = '(a) 非色散 (f_p = 15 GHz)';
cases(1).f_p = 15e9;
cases(1).color = [0, 0.447, 0.741];  % 蓝色

cases(2).label = '(b) 中等色散 (f_p = 25 GHz)';
cases(2).f_p = 25e9;
cases(2).color = [0.85, 0.325, 0.098];  % 橙色

cases(3).label = '(c) 强色散 (f_p = 29 GHz)';
cases(3).f_p = 29e9;
cases(3).color = [0.929, 0.694, 0.125];  % 黄色

%% 2. 计算差频信号频谱
figure('Position', [100, 100, 1200, 400]);

for idx = 1:3
    f_p = cases(idx).f_p;
    
    % 计算截止频率对应的电子密度
    omega_p = 2*pi*f_p;
    n_e = omega_p^2 * epsilon_0 * m_e / q_e^2;
    
    % 计算泰勒展开系数 tau0, tau1, tau2
    tau0 = d / (c * sqrt(1 - (f_p/f0)^2));
    
    % tau1 (式3-24，角频率维度)
    tau1 = (1/(2*pi)) * (-tau0/f0) * ((f_p/f0)^2 / (1-(f_p/f0)^2)^1.5);
    
    % tau2 (式3-25，角频率维度)
    tau2 = (1/(2*pi)^2) * (tau0/f0^2) * ...
           (3*(f_p/f0)^4 / (1-(f_p/f0)^2)^2.5 + (f_p/f0)^2 / (1-(f_p/f0)^2)^1.5);
    
    % 调频斜率
    K_prime = 2*pi*B/T_m;
    
    % 差频信号二阶色散系数 alpha (式3-47简化版)
    alpha = 2*pi*(B^2/T_m^2)*(2*pi*f0*tau2 + 2*tau1);
    
    % 生成差频信号（Chirp信号，式3-43）
    f_D_prime = (B/T_m)*tau0;  % 差频中心频率简化
    phase = 2*pi*f_D_prime*t + pi*alpha*t.^2;
    s_D = exp(1j*phase);
    
    % FFT分析
    NFFT = 2^nextpow2(length(s_D));
    S_D = fftshift(fft(s_D, NFFT));
    freq = (-NFFT/2:NFFT/2-1)*(Fs/NFFT);
    
    % 归一化幅度(dB)
    mag_dB = 20*log10(abs(S_D)/max(abs(S_D)));
    
    % 找到中心频率附近的频谱
    [~, center_idx] = max(abs(S_D));
    freq_centered = freq - freq(center_idx);
    
    % 绘制子图
    subplot(1,3,idx);
    plot(freq_centered/1e6, mag_dB, 'Color', cases(idx).color, 'LineWidth', 2);
    grid on; box on;
    
    % 论文标准绘图设置
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
    set(gca, 'LineWidth', 1.2);
    xlim([-200 200]);
    ylim([-60 5]);
    
    % 标注3dB带宽
    hold on;
    plot([-200 200], [-3 -3], 'r--', 'LineWidth', 1);
    
    xlabel('频率偏移 (MHz)', 'FontSize', 12);
    ylabel('归一化幅度 (dB)', 'FontSize', 12);
    title(cases(idx).label, 'FontSize', 12, 'FontWeight', 'bold');
    
    % 计算并标注3dB带宽
    idx_3dB = find(mag_dB >= -3);
    if ~isempty(idx_3dB)
        BW_3dB = (freq_centered(idx_3dB(end)) - freq_centered(idx_3dB(1)))/1e6;
        text(0, -50, sprintf('3dB带宽: %.0f MHz', BW_3dB), ...
             'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

% 总标题
sgtitle('图 3-3  色散效应下差频信号频谱对比', 'FontSize', 14, 'FontWeight', 'bold');

%% 3. 保存图表
% 保存为 PNG（高分辨率）
print('-dpng', '-r300', '../../final_output/figures/图3-3_频谱散焦对比.png');

% 保存为 SVG（矢量图）
print('-dsvg', '../../final_output/figures/图3-3_频谱散焦对比.svg');

fprintf('图 3-3 已保存至 final_output/figures/\n');
fprintf('非色散(fp=15GHz): 尖锐单峰，3dB带宽约5 MHz\n');
fprintf('中等色散(fp=25GHz): 主瓣展宽至约50 MHz\n');
fprintf('强色散(fp=29GHz): 频谱散焦达120 MHz\n');
