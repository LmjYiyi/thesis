%% plot_fig_3_4.m
% 论文图 3-4：时延演化轨迹对比（无色散 vs 强色散）
% 生成日期：2026-01-06
% 对应章节：3.3.1 群时延的二阶泰勒级数展开与时变时延模型

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
mu = 2*pi*B/T_m;            % 角频率调频斜率 (rad/s^2)

% 等离子体参数
d = 0.15;                   % 等离子体厚度 (m)
nu_e = 1.5e9;               % 碰撞频率 (Hz)

% 情况1：无色散（f_p = 0）
f_p_ideal = 0;
omega_p_ideal = 2*pi*f_p_ideal;

% 情况2：强色散（f_p = 29 GHz）
f_p_strong = 29e9;
omega_p_strong = 2*pi*f_p_strong;

%% 2. 计算时变群时延

% 时间向量
t = linspace(0, T_m, 1000);

% 无色散情况：tau_g(t) = A_0（常数）
A0_ideal = d/c;
tau_g_ideal = A0_ideal * ones(size(t));

% 强色散情况：tau_g(t) = A_0 + A_1*t + A_2*t^2
% 计算展开系数
A0_strong = (d/c) / sqrt(1 - (omega_p_strong/omega_0)^2);

% 一阶色散系数 tau_1
tau_1_strong = -(d/c) * (omega_p_strong^2 / omega_0^3) * ...
               (1 - (omega_p_strong/omega_0)^2)^(-3/2);

% 二阶色散系数 tau_2
tau_2_strong = (3*d/c) * (omega_p_strong^2 / omega_0^4) * ...
               (1 - omega_p_strong^2/(3*omega_0^2)) * ...
               (1 - (omega_p_strong/omega_0)^2)^(-5/2);

% 展开系数
A1_strong = mu * tau_1_strong;
A2_strong = 0.5 * mu^2 * tau_2_strong;

% 时变时延
tau_g_strong = A0_strong + A1_strong*t + A2_strong*t.^2;

%% 3. 绘图

figure('Position', [100, 100, 800, 600]);

% 论文标准颜色
color_ideal = [0.0000, 0.4470, 0.7410];   % 蓝色
color_strong = [0.8500, 0.3250, 0.0980];  % 橙色

% 绘制曲线
plot(t*1e3, tau_g_ideal*1e9, '-', 'Color', color_ideal, 'LineWidth', 2.0, ...
     'DisplayName', '无色散 (f_p = 0)');
hold on;
plot(t*1e3, tau_g_strong*1e9, '--', 'Color', color_strong, 'LineWidth', 2.0, ...
     'DisplayName', sprintf('强色散 (f_p = %.0f GHz)', f_p_strong/1e9));

% 标注关键特征
% 标注线性压缩效应（A_1*t项）
idx_mid = round(length(t)/2);
text(t(idx_mid)*1e3, tau_g_strong(idx_mid)*1e9, ...
     sprintf('  A_1 t 线性压缩\n  (斜率 < 0)'), ...
     'FontSize', 10, 'Color', color_strong);

% 标注高阶畸变（A_2*t^2项）
idx_end = round(length(t)*0.9);
text(t(idx_end)*1e3, tau_g_strong(idx_end)*1e9, ...
     '  A_2 t^2 高阶畸变', ...
     'FontSize', 10, 'Color', color_strong, 'VerticalAlignment', 'top');

% 图表设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
set(gca, 'LineWidth', 1.2);
grid on; box on;

xlabel('时间 t / ms', 'FontSize', 14);
ylabel('群时延 \tau_g / ns', 'FontSize', 14);
title('图 3-4 时延演化轨迹对比', 'FontSize', 14, 'FontWeight', 'bold');

legend('Location', 'northeast', 'FontSize', 11);

%% 4. 保存图表

% 保存为 PNG（高分辨率）
print('-dpng', '-r300', '../figures/图3-4_时延演化轨迹对比.png');

% 保存为 SVG（矢量图）
print('-dsvg', '../figures/图3-4_时延演化轨迹对比.svg');

fprintf('图 3-4 已保存至 final_output/figures/\n');
fprintf('  - 无色散时延: %.3f ns (恒定)\n', A0_ideal*1e9);
fprintf('  - 强色散起始时延: %.3f ns\n', A0_strong*1e9);
fprintf('  - 线性系数 A1: %.3e s^-1\n', A1_strong);
fprintf('  - 二次系数 A2: %.3e s^-2\n', A2_strong);
