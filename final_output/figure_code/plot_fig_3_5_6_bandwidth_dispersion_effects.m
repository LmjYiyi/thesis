%% plot_fig_3_5_6_bandwidth_dispersion_effects.m
% 论文图3-5/图3-6：带宽与色散系数对频谱展宽的影响
% 生成日期：2026-01-24
% 对应章节：3.3.3 频谱特征量化
%
% 图表核心表达：
% - 频谱展宽随带宽 B 的非线性耦合关系（含二次近似）
% - 频谱展宽随色散系数 tau1 / tau2 的变化规律

clear; clc; close all;

%% 1. 物理常数与参数设置（与定稿文档一致）
f0 = 34e9;                     % 起始频率 (Hz)
omega0 = 2*pi*f0;              % 起始角频率 (rad/s)
T_m = 1e-3;                    % 调制周期 (s)
tau0 = 1e-9;                   % 群时延零阶项 (s)，用于展示 C2 的影响

% 色散系数示例（用于对比“加速/减缓”增长趋势）
tau1_cases = [-2, 1] * 1e-12;  % tau1 (s^2) 对应 -2/1 ps/Hz
tau2_cases = [1, 1] * 1e-30;   % tau2 (s^3) 对应 1 fs^2/Hz

% 带宽范围（与正文描述一致）
B = linspace(1e9, 5e9, 300);  % 1-5 GHz

%% 2. 图3-5：频谱展宽随带宽变化
Kp = 2*pi*B/T_m; % 角频率调制斜率

% Case 1：加速增长
tau1 = tau1_cases(1);
tau2 = tau2_cases(1);
C1 = omega0*tau2 + 2*tau1;
C2 = tau1^2 + tau0*tau2;
Delta_f_case1 = (2*pi*B.^2/T_m) .* abs(C1 - (2*pi*B/T_m).*C2);
Delta_f_quad = (2*pi*B.^2/T_m) .* abs(C1); % 二次近似
B_zero_case1 = (C1/C2) * T_m/(2*pi);       % |C1 - K' C2| = 0 对应带宽

% Case 2：减缓增长
tau1 = tau1_cases(2);
tau2 = tau2_cases(2);
C1 = omega0*tau2 + 2*tau1;
C2 = tau1^2 + tau0*tau2;
Delta_f_case2 = (2*pi*B.^2/T_m) .* abs(C1 - (2*pi*B/T_m).*C2);
B_zero_case2 = (C1/C2) * T_m/(2*pi);       % |C1 - K' C2| = 0 对应带宽

figure('Position', [120, 120, 820, 520], 'Color', 'w');
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');

colors = [
    0.0000, 0.4470, 0.7410;  % 蓝
    0.8500, 0.3250, 0.0980;  % 橙
    0.0000, 0.0000, 0.0000;  % 黑
];

plot(B/1e9, Delta_f_case1/1e6, 'o-', 'Color', colors(1,:), 'LineWidth', 1.8, 'MarkerSize', 4);
hold on;
plot(B/1e9, Delta_f_case2/1e6, 's-', 'Color', colors(2,:), 'LineWidth', 1.8, 'MarkerSize', 4);
plot(B/1e9, Delta_f_quad/1e6, '--', 'Color', colors(3,:), 'LineWidth', 2.2);

grid on;
xlabel('带宽 B (GHz)');
ylabel('频谱展宽 \Delta f_D (MHz)');
title('频谱展宽随带宽 B 的变化');
legend( ...
    '\Delta f_D vs B (\tau_1=-2 ps/Hz, \tau_2=1 fs^2/Hz)', ...
    '\Delta f_D vs B (\tau_1=1 ps/Hz, \tau_2=1 fs^2/Hz)', ...
    '二次近似 (\propto B^2)', ...
    'Location', 'northwest');
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

% 标注 |C1 - K' C2| = 0 对应带宽（若落在显示范围内）
if isfinite(B_zero_case1) && B_zero_case1 > min(B) && B_zero_case1 < max(B)
    xline(B_zero_case1/1e9, ':', 'Color', colors(1,:), 'LineWidth', 1.4);
    text(B_zero_case1/1e9 + 0.05, max(Delta_f_case1/1e6)*0.20, ...
        sprintf('B_0^{(1)} = %.2f GHz', B_zero_case1/1e9), ...
        'FontSize', 11, 'Color', colors(1,:));
end
if isfinite(B_zero_case2) && B_zero_case2 > min(B) && B_zero_case2 < max(B)
    xline(B_zero_case2/1e9, ':', 'Color', colors(2,:), 'LineWidth', 1.4);
    text(B_zero_case2/1e9 + 0.05, max(Delta_f_case1/1e6)*0.12, ...
        sprintf('B_0^{(2)} = %.2f GHz', B_zero_case2/1e9), ...
        'FontSize', 11, 'Color', colors(2,:));
end

%% 3. 图3-6：频谱展宽随色散系数变化
B_fixed = 4e9;                     % 固定带宽 (Hz)
Kp_fixed = 2*pi*B_fixed/T_m;

% tau1 变化（tau2 固定）
tau1_range = linspace(-5, 5, 400) * 1e-12; % ps/Hz -> s^2
tau2_fixed = 1e-30;                        % 1 fs^2/Hz -> s^3
C1_tau1 = omega0*tau2_fixed + 2*tau1_range;
C2_tau1 = tau1_range.^2 + tau0*tau2_fixed;
Delta_f_tau1 = (2*pi*B_fixed^2/T_m) .* abs(C1_tau1 - Kp_fixed*C2_tau1);

% tau2 变化（tau1 固定）
tau2_range = linspace(-5, 5, 400) * 1e-30; % fs^2/Hz -> s^3
tau1_fixed = 1e-12;                        % 1 ps/Hz -> s^2
C1_tau2 = omega0*tau2_range + 2*tau1_fixed;
C2_tau2 = tau1_fixed^2 + tau0*tau2_range;
Delta_f_tau2 = (2*pi*B_fixed^2/T_m) .* abs(C1_tau2 - Kp_fixed*C2_tau2);

figure('Position', [980, 120, 820, 520], 'Color', 'w');
plot(tau1_range/1e-12, Delta_f_tau1/1e6, 'LineWidth', 2.0, 'Color', colors(1,:));
hold on;
plot(tau2_range/1e-30, Delta_f_tau2/1e6, 'LineWidth', 2.0, 'Color', colors(2,:));
grid on;
xlabel('\tau_1 (ps/Hz) 或 \tau_2 (fs^2/Hz)');
ylabel('频谱展宽 \Delta f_D (MHz)');
title('频谱展宽随 \tau_1 与 \tau_2 的变化 (B = 4 GHz)');
legend('\Delta f_D vs \tau_1 (\tau_2=1 fs^2/Hz)', '\Delta f_D vs \tau_2 (\tau_1=1 ps/Hz)', ...
    'Location', 'northeast');
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

%% 4. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 图3-5：带宽-散焦非线性耦合关系（与正文图名保持一致）
print('-dpng', '-r300', fullfile(output_dir, '图3-4_带宽散焦耦合曲线.png'));
print('-dsvg', fullfile(output_dir, '图3-4_带宽散焦耦合曲线.svg'));

% 图3-6：色散系数影响
print('-dpng', '-r300', fullfile(output_dir, '图3-6_色散系数影响频谱展宽.png'));
print('-dsvg', fullfile(output_dir, '图3-6_色散系数影响频谱展宽.svg'));

fprintf('✓ 图3-5/图3-6 已保存至 final_output/figures/\\n');
