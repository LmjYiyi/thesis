%% plot_fig_3_5.m
% 论文图 3-5：差频信号频谱散焦效应（不同色散强度对比）
% 生成日期：2026-01-06
% 对应章节：3.3.2 差频信号相位的非线性畸变与瞬时频率解析

clear; clc; close all;

%% 1. 参数设置（与 thesis-code 保持一致）
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% 雷达参数
f0 = 33e9;                  % 起始频率 (Hz)
omega_0 = 2*pi*f0;          % 起始角频率 (rad/s)
B = 3e9;                    % 带宽 (Hz)
T_m = 1e-3;                 % 扫频周期 (s)
K = B/T_m;                  % 频率调频斜率 (Hz/s)
mu = 2*pi*K;                % 角频率调频斜率 (rad/s^2)

% 等离子体参数
d = 0.15;                   % 等离子体厚度 (m)
nu_e = 1.5e9;               % 碰撞频率 (Hz)

% 三种色散情况
f_p_cases = [0, 25e9, 29e9];  % 无色散、弱色散、强色散
case_labels = {'(a) 无色散 (f_p = 0)', ...
               '(b) 弱色散 (f_p = 25 GHz)', ...
               '(c) 强色散 (f_p = 29 GHz)'};

%% 2. 计算差频信号并进行FFT

% 采样参数
fs = 10e6;                  % 采样频率 (Hz)
t = 0:1/fs:T_m;             % 时间向量
N = length(t);              % 采样点数

% 频率向量
f_fft = (-N/2:N/2-1)*(fs/N);

% 存储频谱
spectra = cell(3, 1);

for idx = 1:3
    f_p = f_p_cases(idx);
    omega_p = 2*pi*f_p;
    
    if f_p == 0
        % 无色散：alpha = 0, 差频频率恒定
        tau_0 = d/c;
        f_beat_0 = K * tau_0;
        
        % 差频信号（单频）
        s_beat = exp(1j * 2*pi * f_beat_0 * t);
        
    else
        % 有色散：计算时变时延和alpha
        % 展开系数
        A0 = (d/c) / sqrt(1 - (omega_p/omega_0)^2);
        
        tau_1 = -(d/c) * (omega_p^2 / omega_0^3) * ...
                (1 - (omega_p/omega_0)^2)^(-3/2);
        
        tau_2 = (3*d/c) * (omega_p^2 / omega_0^4) * ...
                (1 - omega_p^2/(3*omega_0^2)) * ...
                (1 - (omega_p/omega_0)^2)^(-5/2);
        
        A1 = mu * tau_1;
        A2 = 0.5 * mu^2 * tau_2;
        
        % 差频中心频率 f'_0（简化计算）
        f_beat_0 = K * A0;
        
        % 二次相位系数 alpha（简化：主要由二阶色散主导）
        alpha = (omega_0 * tau_2 * mu^2) / (2*pi);
        
        % 差频信号相位：phi(t) = 2*pi*f'_0*t + pi*alpha*t^2
        phi = 2*pi*f_beat_0*t + pi*alpha*t.^2;
        s_beat = exp(1j * phi);
    end
    
    % FFT分析
    S_fft = fftshift(fft(s_beat, N));
    S_fft_norm = abs(S_fft) / max(abs(S_fft));  % 归一化
    
    spectra{idx} = S_fft_norm;
end

%% 3. 绘图（三子图）

figure('Position', [100, 100, 1200, 400]);

% 论文标准颜色
colors = [0.0000, 0.4470, 0.7410;   % 蓝色
          0.4660, 0.6740, 0.1880;   % 绿色
          0.8500, 0.3250, 0.0980];  % 橙色

for idx = 1:3
    subplot(1, 3, idx);
    
    % 绘制频谱
    plot(f_fft/1e6, spectra{idx}, 'Color', colors(idx,:), 'LineWidth', 1.5);
    
    % 图表设置
    set(gca, 'FontName', 'SimHei', 'FontSize', 10);
    set(gca, 'LineWidth', 1.0);
    grid on; box on;
    
    xlabel('差频频率 f_D / MHz', 'FontSize', 11, 'FontName', 'SimHei');
    ylabel('归一化幅度', 'FontSize', 11, 'FontName', 'SimHei');
    title(case_labels{idx}, 'FontSize', 11, 'FontName', 'SimHei', 'FontWeight', 'bold');
    
    % 设置合理的X轴范围（聚焦在主瓣附近）
    xlim([0 20]);
    ylim([0 1.1]);
    
    % 标注主瓣宽度（仅对有色散情况）
    if idx > 1
        % 找到3dB带宽
        threshold = 1/sqrt(2);
        idx_peak = find(spectra{idx} == max(spectra{idx}), 1);
        idx_left = find(spectra{idx}(1:idx_peak) < threshold, 1, 'last');
        idx_right = idx_peak + find(spectra{idx}(idx_peak:end) < threshold, 1) - 1;
        
        if ~isempty(idx_left) && ~isempty(idx_right)
            f_left = f_fft(idx_left)/1e6;
            f_right = f_fft(idx_right)/1e6;
            
            % 绘制3dB线
            hold on;
            plot([f_left f_right], [threshold threshold], '--k', 'LineWidth', 0.8);
            text((f_left+f_right)/2, threshold*0.9, ...
                 sprintf('  主瓣宽度\n  ≈%.1f MHz', f_right-f_left), ...
                 'FontSize', 9, 'HorizontalAlignment', 'center');
        end
    end
end

% 总标题
sgtitle('图 3-5 差频信号频谱散焦效应', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');

fprintf('图 3-5 生成完成！\n');
fprintf('三种色散情况的频谱特征：\n');
fprintf('  (a) 无色散: 尖锐单峰\n');
fprintf('  (b) 弱色散 (f_p=25 GHz): 轻微展宽\n');
fprintf('  (c) 强色散 (f_p=29 GHz): 严重散焦\n');
