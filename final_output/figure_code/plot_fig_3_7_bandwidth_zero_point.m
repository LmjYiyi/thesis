%% plot_fig_3_7_bandwidth_zero_point.m
% 论文图3-7：|C1 - K' C2| 的零点（展宽最小点）示意
% 生成日期：2026-01-24
% 对应章节：3.3.3 频谱展宽的非单调性
%
% 图表核心表达：
% - 展示 |C1 - K' C2| 随带宽 B 的变化
% - 标注理论零点 B0（对应展宽最小点/零点）

clear; clc; close all;

%% 1. 参数设置（与前文一致）
f0 = 34e9;                     % 起始频率 (Hz)
omega0 = 2*pi*f0;              % 起始角频率 (rad/s)
T_m = 1e-3;                    % 调制周期 (s)
tau0 = 1e-9;                   % 群时延零阶项 (s)

% 选择一组参数，使零点落在 5 GHz 以内
% 固定 tau2，通过数值反解 tau1 使 B0 落在指定范围
tau2 = 1e-30;                  % tau2 = 1 fs^2/Hz (s^3)
B0_targets = [1, 2, 3, 4, 5] * 1e9; % 目标零点 (Hz)

% 带宽范围
B = linspace(0.2e9, 5.2e9, 400);   % 0.2-5.2 GHz
Kp = 2*pi*B/T_m;

%% 2. 计算 |C1 - K' C2|
% 反解 tau1，使 |C1 - K' C2| = 0 对应 B0
% 方程：B0 = (C1/C2) * T_m/(2*pi)
solve_tau1 = @(B0) fzero( ...
    @(t1) ( (omega0*tau2 + 2*t1) / (t1.^2 + tau0*tau2) ) * T_m/(2*pi) - B0, ...
    1e-12);

tau1_list = arrayfun(solve_tau1, B0_targets);
valid_mask = isfinite(tau1_list);
B0_targets = B0_targets(valid_mask);
tau1_list = tau1_list(valid_mask);

% 计算 |C1 - K' C2|
abs_term_all = zeros(numel(tau1_list), numel(B));
for i = 1:numel(tau1_list)
    t1 = tau1_list(i);
    C1 = omega0*tau2 + 2*t1;
    C2 = t1^2 + tau0*tau2;
    abs_term_all(i, :) = abs(C1 - Kp.*C2);
end

%% 3. 绘图
figure('Position', [120, 140, 820, 520], 'Color', 'w');
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');

colors = lines(numel(tau1_list));
for i = 1:numel(tau1_list)
    plot(B/1e9, abs_term_all(i, :), 'LineWidth', 1.8, 'Color', colors(i, :));
    hold on;
end
grid on;
xlabel('带宽 B (GHz)');
ylabel('|C_1 - K'' C_2|');
title('|C_1 - K'' C_2| 的零点（展宽最小点）');
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

% 标注零点
for i = 1:numel(B0_targets)
    xline(B0_targets(i)/1e9, ':', 'Color', colors(i, :), 'LineWidth', 1.2);
end

% 在图例中补充 tau1/tau2 取值说明
legend_entries = cell(1, numel(B0_targets));
for i = 1:numel(B0_targets)
    legend_entries{i} = sprintf('B_0=%.0f GHz, \\tau_1=%.3f ps/Hz, \\tau_2=%.2f fs^2/Hz', ...
        B0_targets(i)/1e9, tau1_list(i)/1e-12, tau2/1e-30);
end
legend(legend_entries, 'Location', 'northeast');

%% 4. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

print('-dpng', '-r300', fullfile(output_dir, '图3-7_展宽零点示意.png'));
print('-dsvg', fullfile(output_dir, '图3-7_展宽零点示意.svg'));

fprintf('✓ 图3-7 已保存至 final_output/figures/\\n');
