%% plot_fig_3_7.m
% 论文图 3-7：工程判据参数空间映射图
% 生成日期：2026-01-22
% 对应章节：3.4.2 色散效应忽略阈值的理论推导与工程界定
%
% 图表描述（来自定稿文档第158行）：
% "图3-7量化展示了Ka波段诊断场景下'允许带宽B_max'与'截止频率f_p'的参数空间映射关系。
%  判据约束曲线呈现显著的双曲线衰减特征：当截止频率f_p从20 GHz增加至30 GHz时，
%  对应的允许带宽B_max从约8 GHz急剧下降至不足2 GHz，下降幅度超过75%。"

clear; clc; close all;

%% 1. 物理常数与系统参数
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% 雷达参数
f0 = 33e9;                  % 中心频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)
tau_0 = d/c;                % 基础时延 (s)

% 工程判据阈值
C_th = 1;                   % 判据阈值常数

%% 2. 计算允许带宽 B_max 随截止频率 f_p 的变化

% 截止频率范围：15-32 GHz
f_p_range = linspace(15e9, 32e9, 500);

% 初始化允许带宽数组
B_max = zeros(size(f_p_range));

% 对每个截止频率计算允许的最大带宽
for i = 1:length(f_p_range)
    f_p = f_p_range(i);
    
    % 计算非线性度因子 η(f0)
    % η = (B/f0) * (f_p/f0)^2 / [1-(f_p/f0)^2]^(3/2)
    % 为了求 B_max，需要反解：B * η * τ_0 = C_th
    % 即：B * (B/f0) * (f_p/f0)^2 / [1-(f_p/f0)^2]^(3/2) * τ_0 = C_th
    % 简化：B^2 * (f_p/f0)^2 / [f0 * [1-(f_p/f0)^2]^(3/2)] * τ_0 = C_th
    
    % 使用简化形式：η ≈ (B/f0) * (f_p/f0)^2 / [1-(f_p/f0)^2]^(3/2)
    % 对于固定的 f_p，η 与 B 成正比
    % 因此 B_max = C_th / (η_unit * τ_0)
    % 其中 η_unit 是 B=1 Hz 时的 η 值
    
    ratio = f_p / f0;
    
    % 避免接近截止频率时的奇异性
    if ratio >= 0.95
        B_max(i) = NaN;  % 超出有效范围
        continue;
    end
    
    % η 的单位值（B=1 Hz时）
    eta_unit = (1/f0) * ratio^2 / (1 - ratio^2)^(3/2);
    
    % 允许的最大带宽
    B_max(i) = C_th / (eta_unit * tau_0);
end

%% 3. 绘制参数空间映射图

figure('Position', [100, 100, 900, 700]);

% 绘制判据约束曲线
plot(f_p_range/1e9, B_max/1e9, 'LineWidth', 2.5, 'Color', [0.0000, 0.4470, 0.7410]);
hold on; grid on; box on;

% 标注典型工况点（f_p = 29 GHz, B = 3 GHz）
f_p_typical = 29e9;
B_typical = 3e9;
plot(f_p_typical/1e9, B_typical/1e9, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
text(f_p_typical/1e9 + 0.5, B_typical/1e9 + 0.3, ...
    sprintf('典型工况\n($f_p$ = 29 GHz, $B$ = 3 GHz)'), ...
    'FontSize', 11, 'Color', 'r', 'Interpreter', 'latex');

% 添加关键数据点标注（验证文档描述）
% f_p = 20 GHz → B_max ≈ 8 GHz
idx_20 = find(abs(f_p_range - 20e9) < 0.1e9, 1);
if ~isempty(idx_20)
    plot(f_p_range(idx_20)/1e9, B_max(idx_20)/1e9, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    text(f_p_range(idx_20)/1e9 - 1.5, B_max(idx_20)/1e9, ...
        sprintf('$B_{max}$ ≈ %.1f GHz', B_max(idx_20)/1e9), ...
        'FontSize', 10, 'Interpreter', 'latex');
end

% f_p = 30 GHz → B_max ≈ 2 GHz
idx_30 = find(abs(f_p_range - 30e9) < 0.1e9, 1);
if ~isempty(idx_30)
    plot(f_p_range(idx_30)/1e9, B_max(idx_30)/1e9, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    text(f_p_range(idx_30)/1e9 + 0.3, B_max(idx_30)/1e9, ...
        sprintf('$B_{max}$ ≈ %.1f GHz', B_max(idx_30)/1e9), ...
        'FontSize', 10, 'Interpreter', 'latex');
end

% 添加区域着色（可选）
% 强色散区（判据被突破）
fill([f_p_range/1e9, fliplr(f_p_range/1e9)], ...
     [B_max/1e9, ones(size(B_max))*10], ...
     [1, 0.8, 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text(25, 7, '强色散区', 'FontSize', 12, 'Color', [0.8, 0, 0], 'FontWeight', 'bold');
text(25, 6.3, '($B \cdot \eta \cdot \tau_0 > 1$)', 'FontSize', 10, 'Color', [0.8, 0, 0], 'Interpreter', 'latex');

% 弱色散区（判据满足）
text(18, 2, '弱色散区', 'FontSize', 12, 'Color', [0, 0.6, 0], 'FontWeight', 'bold');
text(18, 1.3, '($B \cdot \eta \cdot \tau_0 < 1$)', 'FontSize', 10, 'Color', [0, 0.6, 0], 'Interpreter', 'latex');

% 坐标轴设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);
set(gca, 'LineWidth', 1.2);
xlabel('截止频率 $f_p$ / GHz', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('允许带宽 $B_{max}$ / GHz', 'FontSize', 15, 'Interpreter', 'latex');
title('图 3-7 工程判据参数空间映射：允许带宽与截止频率的约束关系', ...
      'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

% 设置坐标轴范围
xlim([15, 32]);
ylim([0, 10]);

% 添加图例
legend('判据约束曲线 ($B \cdot \eta \cdot \tau_0 = 1$)', ...
       '典型工况点', ...
       'Location', 'northeast', 'FontSize', 11, 'Interpreter', 'latex');

%% 4. 保存图表

% 确保输出目录存在
if ~exist('../../final_output/figures', 'dir')
    mkdir('../../final_output/figures');
end

% 保存为 PNG（高分辨率）
print('-dpng', '-r300', '../../final_output/figures/图3-7_工程判据参数空间映射.png');

% 保存为 SVG（矢量图，用于排版）
print('-dsvg', '../../final_output/figures/图3-7_工程判据参数空间映射.svg');

fprintf('图 3-7 已保存至 final_output/figures/\n');
fprintf('验证：f_p = 20 GHz → B_max ≈ %.2f GHz\n', B_max(idx_20)/1e9);
fprintf('验证：f_p = 30 GHz → B_max ≈ %.2f GHz\n', B_max(idx_30)/1e9);
fprintf('下降幅度：%.1f%%\n', (1 - B_max(idx_30)/B_max(idx_20)) * 100);
