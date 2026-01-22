%% plot_fig_3_7_dispersion_criterion.m
% 论文图 3-7：色散效应工程判据参数空间图（不可能三角）
% 生成日期：2026-01-22
% 对应章节：3.4.2 色散效应忽略阈值的工程界定
%
% 图表描述（from final.md）：
% "判据约束曲线呈现显著的双曲线衰减特征：当截止频率 fp 从 20 GHz 增加至 30 GHz 时，
%  对应的允许带宽 Bmax 从约 8 GHz 急剧下降至不足 2 GHz"
%
% 物理意义：
% - 绿色区域（曲线下方）：传统FFT方法适用区（ξ < 1）
% - 红色区域（曲线上方）：高级算法必要区（ξ > 1）
% - 工况A（fp=29 GHz）：安全点标注
% - 工况B（fp=32 GHz）：失效点标注

clear; clc; close all;

%% 1. 物理常数与雷达参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 雷达参数（固定）
d = 0.15;                   % 等离子体厚度 (m)
tau_0 = d / c;              % 基础时延 (s)
f_c = 35.5e9;               % 中心频率 (Hz)

%% 2. 截止频率扫描范围
fp_range = linspace(20e9, 33e9, 500);  % 20-33 GHz

% 计算每个 fp 对应的允许最大带宽 Bmax
Bmax_values = zeros(size(fp_range));

for i = 1:length(fp_range)
    fp = fp_range(i);
    
    % 计算归一化频率比 x = fp / fc
    x = fp / f_c;
    
    % 计算非线性度因子 η(f_c) （式3-18）
    % η = (B/f) * (fp/f)^2 / [1-(fp/f)^2]^(3/2)
    % 这里用单位带宽 B=1 计算，后续求解 Bmax
    
    if x < 1  % 只在透射区计算
        denominator = (1 - x^2)^1.5;
        eta_per_B = (1/f_c) * (x^2 / denominator);  % η/B
        
        % 由判据 B * η * τ_0 <= 1
        % Bmax = 1 / (η_per_B * B * τ_0) = 1 / (η_per_B * τ_0)
        Bmax_values(i) = 1 / (eta_per_B * tau_0);
    else
        Bmax_values(i) = NaN;  % 超过截止频率
    end
end

%% 3. 标注工况点
% 工况A：安全区（fp = 29 GHz, B = 3 GHz, ξ = 0.42）
fpA = 29e9;
BA = 3e9;
xiA = 0.42;

% 工况B：失效区（fp = 32 GHz, B = 3 GHz, ξ = 1.275）
fpB = 32e9;
BB = 3e9;
xiB = 1.275;

%% 4. 绘图
figure('Position', [100, 100, 900, 700], 'Color', 'w');

% 填充区域（安全区 vs 危险区）
hold on; box on; grid on;

% 绿色区域：传统方法适用区（曲线下方）
fill([fp_range/1e9, fliplr(fp_range/1e9)], ...
     [zeros(size(Bmax_values)), fliplr(Bmax_values/1e9)], ...
     [0.85, 0.95, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

% 红色区域：高级算法必要区（曲线上方）
y_max = 10;  % Y轴上限
fill([fp_range/1e9, fliplr(fp_range/1e9)], ...
     [Bmax_values/1e9, y_max*ones(size(Bmax_values))], ...
     [0.95, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

% 判据临界曲线（B * η * τ_0 = 1）
plot(fp_range/1e9, Bmax_values/1e9, 'k-', 'LineWidth', 2.5, ...
    'DisplayName', '判据临界线 (B·η·τ_0 = 1)');

% 工况A标注（安全点）
plot(fpA/1e9, BA/1e9, 'go', 'MarkerSize', 14, 'MarkerFaceColor', 'g', ...
    'LineWidth', 2, 'DisplayName', sprintf('工况A: f_p=%.0f GHz (ξ=%.2f)', fpA/1e9, xiA));
text(fpA/1e9 - 1.5, BA/1e9 + 0.5, sprintf('  安全区\n  ξ=%.2f<1', xiA), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [0, 0.6, 0]);

% 工况B标注（失效点）
plot(fpB/1e9, BB/1e9, 'ro', 'MarkerSize', 14, 'MarkerFaceColor', 'r', ...
    'LineWidth', 2, 'DisplayName', sprintf('工况B: f_p=%.0f GHz (ξ=%.2f)', fpB/1e9, xiB));
text(fpB/1e9 + 0.3, BB/1e9 + 0.5, sprintf('失效区\nξ=%.2f>1', xiB), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8, 0, 0]);

% 区域标签
text(25, 7, '传统FFT方法适用', 'FontSize', 14, 'Color', [0, 0.5, 0], ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(28, 1, '高级算法必要区', 'FontSize', 14, 'Color', [0.7, 0, 0], ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', -35);

%% 5. 坐标轴与标签
xlabel('等离子体截止频率 f_p (GHz)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('允许最大带宽 B_{max} (GHz)', 'FontSize', 14, 'FontWeight', 'bold');
title('图 3-7 色散效应工程判据的参数空间约束（不可能三角）', ...
    'FontSize', 15, 'FontWeight', 'bold');

% 设置坐标范围
xlim([20, 33]);
ylim([0, y_max]);

% 图例
legend('Location', 'northeast', 'FontSize', 11, 'Box', 'on');

% 字体设置
set(gca, 'FontName', 'SimHei', 'FontSize', 12, 'LineWidth', 1.2);
grid on;

%% 6. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率）
print('-dpng', '-r300', fullfile(output_dir, '图3-7_色散效应判据参数空间.png'));

% 保存SVG（矢量图）
print('-dsvg', fullfile(output_dir, '图3-7_色散效应判据参数空间.svg'));

fprintf('✓ 图 3-7 已保存至 final_output/figures/\n');
fprintf('  - PNG: 300 DPI (用于预览)\n');
fprintf('  - SVG: 矢量图 (用于LaTeX排版)\n');
