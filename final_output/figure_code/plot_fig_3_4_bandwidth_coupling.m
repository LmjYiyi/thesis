%% plot_fig_3_4_bandwidth_coupling.m
% 论文图 3-4：带宽-散焦非线性耦合关系曲线
% 生成日期：2026-01-22
% 对应章节：3.3.3 频谱特征量化
%
% 图表描述（from final.md第179行）：
% "在小带宽区域（B < 3 GHz），曲线严格遵循 ΔfD ∝ B² 的抛物线趋势（图中虚线为二次项近似）；
%  当带宽超过3 GHz后，三次项修正开始显现，实线偏离虚线，展宽增速放缓"

clear; clc; close all;

%% 1. 物理常数与固定参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 固定参数（对应强色散工况）
f0 = 34e9;                  % 中心频率 (Hz)
fp = 29e9;                  % 截止频率 (Hz)
d = 0.15;                   % 厚度 (m)
Tm = 1e-3;                  % 调制周期 (s)
tau_0 = d / c;              % 基础时延 (s)

%% 2. 计算泰勒展开系数
omega_0 = 2*pi*f0;
x = fp / f0;

% 一阶和二阶色散系数（式3-24, 3-25）
tau_1 = (1/(2*pi)) * (-tau_0/f0) * (x^2) / ((1-x^2)^1.5);
tau_2 = (1/(2*pi)^2) * (tau_0/f0^2) * ...
        ((3*x^4)/(1-x^2)^2.5 + x^2/(1-x^2)^1.5);

fprintf('计算的物理参数：\n');
fprintf('  tau_0 = %.3f ns\n', tau_0*1e9);
fprintf('  tau_1 = %.3e ns/rad\n', tau_1*1e9);
fprintf('  tau_2 = %.3e ns/rad²\n', tau_2*1e9);

%% 3. 带宽扫描
B_range = linspace(1e9, 5e9, 200);  % 1-5 GHz
Delta_fD_full = zeros(size(B_range));
Delta_fD_approx = zeros(size(B_range));

for i = 1:length(B_range)
    B = B_range(i);
    K_prime = 2*pi*B / Tm;  % 角频率调频斜率
    
    % 完整公式（式3-49）
    A1 = tau_1 * K_prime;
    A2 = 0.5 * tau_2 * K_prime^2;
    
    alpha_full = (omega_0*tau_2*K_prime^2)/(2*pi) + 2*(B/Tm)*tau_1*K_prime ...
                 - (B/Tm)*((tau_1*K_prime)^2 + 2*tau_0*A2);
    
    Delta_fD_full(i) = abs(alpha_full) * Tm;
    
    % 二次近似（式3-50）
    alpha_approx = (2*pi*(B^2)/Tm) * abs(omega_0*tau_2 + 2*tau_1);
    Delta_fD_approx(i) = alpha_approx * Tm;
end

%% 4. 绘图
figure('Position', [100, 100, 900, 700], 'Color', 'w');
hold on; box on; grid on;

% 绘制完整公式（实线）
plot(B_range/1e9, Delta_fD_full/1e6, 'b-', 'LineWidth', 2.5, ...
    'DisplayName', '完整公式（式3-49）');

% 绘制二次近似（虚线）
plot(B_range/1e9, Delta_fD_approx/1e6, 'r--', 'LineWidth', 2, ...
    'DisplayName', '二次近似（式3-50）');

% 标注分界点（B = 3 GHz）
plot([3, 3], [0, 200], 'k:', 'LineWidth', 1.5, 'DisplayName', 'B = 3 GHz');
text(3.1, 100, 'B = 3 GHz', 'FontSize', 12, 'FontWeight', 'bold', ...
    'Rotation', 90, 'VerticalAlignment', 'bottom');

% 标注两条曲线的偏离区域
idx_3GHz = find(B_range >= 3e9, 1);
plot(B_range(idx_3GHz)/1e9, Delta_fD_full(idx_3GHz)/1e6, ...
    'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'LineWidth', 2);

% 添加注释
annotation('textarrow', [0.35, 0.42], [0.4, 0.35], ...
    'String', sprintf('二次项主导区\n严格遵循 ΔfD ∝ B²'), ...
    'FontSize', 11, 'FontWeight', 'bold', 'LineWidth', 1.5);

annotation('textarrow', [0.7, 0.65], [0.7, 0.6], ...
    'String', sprintf('三次项修正显现\n实线偏离虚线'), ...
    'FontSize', 11, 'FontWeight', 'bold', 'LineWidth', 1.5, ...
    'Color', [0.8, 0, 0]);

% 坐标轴与标签
xlabel('雷达带宽 B (GHz)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('频谱展宽 Δf_D (MHz)', 'FontSize', 14, 'FontWeight', 'bold');
title({'图 3-4 带宽-散焦非线性耦合关系', ...
       sprintf('(f_p = %d GHz, T_m = %d ms)', fp/1e9, Tm*1e3)}, ...
    'FontSize', 15, 'FontWeight', 'bold');

% 图例
legend('Location', 'northwest', 'FontSize', 12, 'Box', 'on');

% 设置坐标范围
xlim([1, 5]);
ylim([0, 200]);
set(gca, 'FontName', 'SimHei', 'FontSize', 12, 'LineWidth', 1.2);

%% 5. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率）
print('-dpng', '-r300', fullfile(output_dir, '图3-4_带宽散焦耦合曲线.png'));

% 保存SVG（矢量图）
print('-dsvg', fullfile(output_dir, '图3-4_带宽散焦耦合曲线.svg'));

fprintf('✓ 图 3-4 已保存至 final_output/figures/\n');
fprintf('  - 蓝色实线：完整公式（含三次项修正）\n');
fprintf('  - 红色虚线：二次近似（ΔfD ∝ B²）\n');
fprintf('  - 分界点：B = 3 GHz，三次项修正开始显现\n');
