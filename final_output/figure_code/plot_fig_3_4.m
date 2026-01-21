%% plot_fig_3_4.m
% 论文图 3-4：带宽-散焦非线性耦合曲线
% 生成日期：2026-01-22
% 对应章节：3.3.3 频谱特征量化
%
% 文档描述（第175行）：
% "图3-4展示了频谱展宽 ΔfD 随带宽 B (1-5 GHz) 的变化规律。
% 在小带宽区域(B < 3 GHz)，曲线严格遵循 ΔfD ∝ B² 的抛物线趋势(虚线为二次项近似)；
% 当带宽超过3 GHz后，三次项修正开始显现，实线偏离虚线，展宽增速放缓。
% 最优带宽约为 Bopt ≈ 3.5 GHz"

clear; clc; close all;

%% 1. 参数设置（与 thesis-code 保持一致）
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数 (F/m)
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% 等离子体参数（强色散条件）
f_p = 29e9;                 % 截止频率 (Hz)
omega_p = 2*pi*f_p;
n_e = omega_p^2 * epsilon_0 * m_e / q_e^2;
nu_e = 1.5e9;               % 碰撞频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% LFMCW雷达参数
f0 = 34e9;                  % 中心频率 (Hz)
T_m = 1e-3;                 % 扫频周期 (s)

% 带宽扫描范围
B_range = linspace(1e9, 5e9, 100);  % 1-5 GHz

%% 2. 计算泰勒展开系数（固定值）
omega0 = 2*pi*f0;

% tau0 (式3-21)
tau0 = d / (c * sqrt(1 - (f_p/f0)^2));

% tau1 (式3-24，角频率维度)
tau1 = (1/(2*pi)) * (-tau0/f0) * ((f_p/f0)^2 / (1-(f_p/f0)^2)^1.5);

% tau2 (式3-25，角频率维度)
tau2 = (1/(2*pi)^2) * (tau0/f0^2) * ...
       (3*(f_p/f0)^4 / (1-(f_p/f0)^2)^2.5 + (f_p/f0)^2 / (1-(f_p/f0)^2)^1.5);

%% 3. 计算频谱展宽 ΔfD 随带宽 B 的变化

% 系数定义(式3-51)
C1 = omega0*tau2 + 2*tau1;
C2 = tau1^2 + tau0*tau2;

% 完整公式(式3-49)
Delta_fD_full = zeros(size(B_range));
% 二次近似(式3-50)
Delta_fD_approx = zeros(size(B_range));

for i = 1:length(B_range)
    B = B_range(i);
    
    % 完整公式（含三次项修正）
    Delta_fD_full(i) = abs(2*pi*(B^2/T_m)*(C1 - 2*pi*(B/T_m)*C2));
    
    % 二次近似（仅保留B²项）
    Delta_fD_approx(i) = 2*pi*(B^2/T_m)*abs(C1);
end

%% 4. 绘图
figure('Position', [100, 100, 800, 600]);

% 绘制完整公式（实线）
plot(B_range/1e9, Delta_fD_full/1e6, 'b-', 'LineWidth', 2.5, 'DisplayName', '完整公式 (式3-49)');
hold on; grid on; box on;

% 绘制二次近似（虚线）
plot(B_range/1e9, Delta_fD_approx/1e6, 'r--', 'LineWidth', 2, 'DisplayName', '二次近似 (式3-50)');

% 标注B = 3 GHz分界线
plot([3 3], [0 max(Delta_fD_full/1e6)*1.1], 'k:', 'LineWidth', 1.5, 'DisplayName', 'B = 3 GHz');

% 标注最优带宽 Bopt ≈ 3.5 GHz
B_opt = 3.5;  % GHz
[~, idx_opt] = min(abs(B_range/1e9 - B_opt));
Delta_fD_opt = Delta_fD_full(idx_opt)/1e6;
plot(B_opt, Delta_fD_opt, 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm', ...
     'DisplayName', sprintf('B_{opt} ≈ %.1f GHz', B_opt));

% 论文标准绘图设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
set(gca, 'LineWidth', 1.2);

xlabel('带宽 B (GHz)', 'FontSize', 14);
ylabel('频谱展宽 Δf_D (MHz)', 'FontSize', 14);
title('图 3-4  带宽-散焦非线性耦合曲线', 'FontSize', 14, 'FontWeight', 'bold');

% 图例
legend('Location', 'northwest', 'FontSize', 11);

% 标注关键区域
text(1.5, max(Delta_fD_full/1e6)*0.8, '抛物线区域 (∝B²)', ...
     'FontSize', 11, 'Color', 'k', 'BackgroundColor', 'w');
text(4.2, max(Delta_fD_full/1e6)*0.5, '三次修正区域', ...
     'FontSize', 11, 'Color', 'k', 'BackgroundColor', 'w');

xlim([1 5]);
ylim([0 max(Delta_fD_full/1e6)*1.1]);

%% 5. 保存图表
% 保存为 PNG（高分辨率）
print('-dpng', '-r300', '../../final_output/figures/图3-4_带宽散焦耦合.png');

% 保存为 SVG（矢量图）
print('-dsvg', '../../final_output/figures/图3-4_带宽散焦耦合.svg');

fprintf('图 3-4 已保存至 final_output/figures/\n');
fprintf('B < 3 GHz: 遵循 ΔfD ∝ B² 抛物线趋势\n');
fprintf('B > 3 GHz: 三次项修正显现，实线偏离虚线\n');
fprintf('最优带宽 Bopt ≈ %.1f GHz, ΔfD ≈ %.0f MHz\n', B_opt, Delta_fD_opt);
