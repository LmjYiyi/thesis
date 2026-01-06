%% plot_fig_3_6.m
% 论文图 3-6：带宽-散焦非线性耦合曲线
% 生成日期：2026-01-06
% 对应章节：3.3.4 频谱特征量化：二阶色散导致的散焦效应与带宽耦合机制

clear; clc; close all;

%% 1. 参数设置（与 thesis-code 保持一致）
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% 雷达参数
f0 = 33e9;                  % 起始频率 (Hz)
omega_0 = 2*pi*f0;          % 起始角频率 (rad/s)
T_m = 1e-3;                 % 扫频周期 (s)

% 等离子体参数（强色散情况）
d = 0.15;                   % 等离子体厚度 (m)
f_p = 29e9;                 % 截止频率 (Hz)
omega_p = 2*pi*f_p;
nu_e = 1.5e9;               % 碰撞频率 (Hz)

%% 2. 计算色散系数

% 基础群时延
tau_0 = (d/c) / sqrt(1 - (omega_p/omega_0)^2);

% 一阶色散系数 tau_1
tau_1 = -(d/c) * (omega_p^2 / omega_0^3) * ...
        (1 - (omega_p/omega_0)^2)^(-3/2);

% 二阶色散系数 tau_2
tau_2 = (3*d/c) * (omega_p^2 / omega_0^4) * ...
        (1 - omega_p^2/(3*omega_0^2)) * ...
        (1 - (omega_p/omega_0)^2)^(-5/2);

% 定义色散系数 C1 和 C2
C1 = omega_0 * tau_2 + 2*tau_1;
C2 = tau_1^2 + tau_0 * tau_2;

%% 3. 计算频谱展宽随带宽的变化

% 带宽范围：1-5 GHz
B_range = linspace(1e9, 5e9, 100);

% 频谱展宽 Delta_f_D（完整公式）
Delta_f_D_full = zeros(size(B_range));

for i = 1:length(B_range)
    B = B_range(i);
    % 式 (3-52)
    Delta_f_D_full(i) = (2*pi*B^2/T_m) * abs(C1 - 2*pi*(B/T_m)*C2);
end

% 二次近似（小带宽）
Delta_f_D_quad = (2*pi*B_range.^2/T_m) * abs(C1);

%% 4. 绘图

figure('Position', [100, 100, 800, 600]);

% 论文标准颜色
color_full = [0.8500, 0.3250, 0.0980];    % 橙色 - 完整公式
color_quad = [0.0000, 0.4470, 0.7410];    % 蓝色 - 二次近似

% 绘制完整公式曲线
plot(B_range/1e9, Delta_f_D_full/1e6, '-', 'Color', color_full, ...
     'LineWidth', 2.5, 'DisplayName', '完整公式 (式 3-52)');
hold on;

% 绘制二次近似曲线
plot(B_range/1e9, Delta_f_D_quad/1e6, '--', 'Color', color_quad, ...
     'LineWidth', 2.0, 'DisplayName', '二次近似 (\Delta f_D \propto B^2)');

% 标注关键区域
% 二次主导区
idx_quad = find(B_range <= 2.5e9, 1, 'last');
text(B_range(idx_quad)/1e9, Delta_f_D_full(idx_quad)/1e6, ...
     '  二次主导区\n  (\Delta f_D \propto B^2)', ...
     'FontSize', 11, 'Color', color_quad, 'VerticalAlignment', 'bottom');

% 三次修正区
idx_cubic = find(B_range >= 4e9, 1);
text(B_range(idx_cubic)/1e9, Delta_f_D_full(idx_cubic)/1e6, ...
     '  三次修正区\n  (偏离 B^2 关系)  ', ...
     'FontSize', 11, 'Color', color_full, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

% 标注典型工作点（B = 3 GHz）
B_typical = 3e9;
idx_typical = find(B_range >= B_typical, 1);
Delta_f_typical = Delta_f_D_full(idx_typical);

plot(B_typical/1e9, Delta_f_typical/1e6, 'o', ...
     'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', ...
     'LineWidth', 1.5, 'DisplayName', sprintf('工作点 (B = %.0f GHz)', B_typical/1e9));

text(B_typical/1e9, Delta_f_typical/1e6, ...
     sprintf('  B = %.0f GHz\n  \\Delta f_D ≈ %.0f MHz', ...
             B_typical/1e9, Delta_f_typical/1e6), ...
     'FontSize', 10, 'VerticalAlignment', 'bottom');

% 图表设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
set(gca, 'LineWidth', 1.2);
grid on; box on;

xlabel('带宽 B / GHz', 'FontSize', 14);
ylabel('频谱展宽 \Delta f_D / MHz', 'FontSize', 14);
title('图 3-6 带宽-散焦非线性耦合曲线', 'FontSize', 14, 'FontWeight', 'bold');

legend('Location', 'northwest', 'FontSize', 11);

% 设置坐标轴范围
xlim([1 5]);
ylim([0 max(Delta_f_D_full/1e6)*1.1]);

%% 5. 保存图表

% 保存为 PNG（高分辨率）
print('-dpng', '-r300', '../figures/图3-6_带宽散焦耦合曲线.png');

% 保存为 SVG（矢量图）
print('-dsvg', '../figures/图3-6_带宽散焦耦合曲线.svg');

fprintf('图 3-6 已保存至 final_output/figures/\n');
fprintf('关键参数：\n');
fprintf('  - 截止频率 f_p: %.0f GHz\n', f_p/1e9);
fprintf('  - 色散系数 C1: %.3e\n', C1);
fprintf('  - 色散系数 C2: %.3e\n', C2);
fprintf('  - 典型工作点 (B=3 GHz): \Delta f_D ≈ %.0f MHz\n', Delta_f_typical/1e6);
fprintf('\n物理意义：\n');
fprintf('  - 小带宽区: \Delta f_D 严格遵循 B^2 关系（抛物线）\n');
fprintf('  - 大带宽区: 三次项修正显现，偏离 B^2 趋势\n');
fprintf('  - 工程悖论: 带宽增加提高分辨率，但也放大散焦效应\n');
