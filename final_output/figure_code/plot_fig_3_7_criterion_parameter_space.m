%% plot_fig_3_7_criterion_parameter_space.m
% 图3-7：色散效应工程判据的参数空间约束（不可能三角）
% 优化目标：博士论文出版级画质（LaTeX字体、矢量配色、精细标注）
% 生成日期：2026-01-24
% 对应章节：3.4.2 色散效应忽略阈值的理论推导与工程界定

clear; clc; close all;

%% 1. 全局绘图参数设置（Publication Standards）
fig_width = 16;                 % cm
fig_height = 10;                % cm
font_en = 'Times New Roman';    % 英文字体
font_cn = 'SimHei';             % 中文字体
font_size_base = 10;            % 基础字号

% 颜色定义（RGB 归一化）
color_curve = [0, 60, 136]/255;     % 深蓝 - 主曲线
color_safe = [235, 245, 235]/255;   % 淡绿 - 安全区
color_fail = [250, 235, 235]/255;   % 淡红 - 失效区
color_typ = [200, 30, 30]/255;      % 深红 - 典型工况点
color_text = [0.15, 0.15, 0.15];    % 深灰 - 文字

%% 2. 物理模型计算
c = 3e8;
f0 = 34e9;
d = 0.15;
tau_0 = d/c;
eta_th = 1;

f_p_range = linspace(18e9, 37e9, 1200); % 扩展范围以平滑曲线
B_max = nan(size(f_p_range));

for i = 1:numel(f_p_range)
    f_p = f_p_range(i);
    ratio = f_p / f0;
    if ratio >= 0.9995
        continue;
    end

    % eta 线性化单位值（B = 1 Hz 时）
    eta_unit = (1/f0) * ratio^2 / (1 - ratio^2)^(3/2);

    % eta 与带宽成正比：B^2 * eta_unit * tau_0 = 1
    B_max(i) = sqrt(eta_th / (eta_unit * tau_0));
end

% 数据清洗（去除 NaN 与无穷）
valid_idx = ~isnan(B_max) & ~isinf(B_max);
f_p_plot = f_p_range(valid_idx) / 1e9;
B_max_plot = B_max(valid_idx) / 1e9;

%% 2.1 两种工况的 ξ 值计算与打印
B_sys_Hz = 3e9;   % 系统带宽 (Hz)，与图中 B_sys = 3 GHz 一致
f_p_A = 29e9;     % Case A 截止频率 (Hz)
f_p_B = 33.5e9;   % Case B 截止频率 (Hz)
ratio_A = f_p_A / f0;
ratio_B = f_p_B / f0;
eta_unit_A = (1/f0) * ratio_A^2 / (1 - ratio_A^2)^(3/2);
eta_unit_B = (1/f0) * ratio_B^2 / (1 - ratio_B^2)^(3/2);
eta_A = B_sys_Hz * eta_unit_A;
eta_B = B_sys_Hz * eta_unit_B;
xi_A = B_sys_Hz * eta_A * tau_0;
xi_B = B_sys_Hz * eta_B * tau_0;

fprintf('工况 A (f_p = 29 GHz, B = 3 GHz): ξ_A = B·η·τ_0 = %.4f\n', xi_A);
fprintf('工况 B (f_p = 33.5 GHz, B = 3 GHz): ξ_B = B·η·τ_0 = %.4f\n', xi_B);

%% 3. 绘图层级构建
figure('Units', 'centimeters', 'Position', [5, 5, fig_width, fig_height], 'Color', 'w');
ax = gca;
hold on;

% --- A. 绘制背景区域 ---
y_limit_max = 7.2; 
x_limit_min = 25;
x_limit_max = 34;

% 裁剪到显示区间，避免填充区域出现空白
in_view = f_p_plot >= x_limit_min & f_p_plot <= x_limit_max;
f_p_view = f_p_plot(in_view);
B_view = B_max_plot(in_view);

% 安全区（下部）
x_poly_safe = [x_limit_min, f_p_view, x_limit_max, x_limit_max, x_limit_min];
y_poly_safe = [0, B_view, B_view(end), 0, 0];
fill(x_poly_safe, y_poly_safe, color_safe, 'EdgeColor', 'none', 'FaceAlpha', 1);

% 失效区（上部）
x_poly_fail = [x_limit_min, f_p_view, x_limit_max, x_limit_max, x_limit_min];
y_poly_fail = [y_limit_max, B_view, B_view(end), y_limit_max, y_limit_max];
fill(x_poly_fail, y_poly_fail, color_fail, 'EdgeColor', 'none', 'FaceAlpha', 1);

% --- B. 网格与边框 ---
grid on;
set(ax, 'Layer', 'top', ...
    'GridColor', [0.4, 0.4, 0.4], ...
    'GridAlpha', 0.15, ...
    'LineWidth', 1.0, ...
    'TickDir', 'out', ...
    'FontName', font_en, ...
    'FontSize', font_size_base, ...
    'XMinorTick', 'on', 'YMinorTick', 'on');

% --- C. 主曲线 ---
plot(f_p_plot, B_max_plot, 'LineWidth', 2.0, 'Color', color_curve);

% --- D. 关键数据点 ---
% （已移除 f_p=20/30 GHz 的通用标记，改为针对 Case B 的临界点标注）

% 系统带宽参考线（贯穿全图）
B_sys = 3;
yline(B_sys, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 0.8);
text(x_limit_min + 0.2, B_sys + 0.25, '系统带宽 B = 3 GHz', ...
    'FontName', font_cn, 'FontSize', 8, 'Color', [0.45, 0.45, 0.45]);

% 典型工况点（29 GHz, 3 GHz）- 安全区内
f_typ = 29; B_typ = B_sys;
plot([f_typ, f_typ], [0, B_typ], '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 0.8);
h_pt_safe = plot(f_typ, B_typ, 'o', 'MarkerSize', 7, ...
    'MarkerFaceColor', color_typ, 'MarkerEdgeColor', 'none');
text(f_typ - 0.3, B_typ - 0.4, 'Case A', ...
    'FontName', font_en, 'FontSize', 9, 'FontWeight', 'bold', ...
    'Color', color_typ, 'HorizontalAlignment', 'right');

% 失效工况点（33.5 GHz, 3 GHz）- 失效区内
f_fail = 33.5; B_fail = B_sys;
plot([f_fail, f_fail], [0, B_fail], '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 0.8);
h_pt_fail = plot(f_fail, B_fail, 'v', 'MarkerSize', 7, ...
    'MarkerFaceColor', color_curve, 'MarkerEdgeColor', 'none');
text(f_fail - 0.2, B_fail + 0.6, 'Case B', ...
    'FontName', font_en, 'FontSize', 9, 'FontWeight', 'bold', ...
    'Color', color_curve, 'HorizontalAlignment', 'right');

% Case B 对应的允许最大带宽（判据曲线交点）
B_fail_max = interp1(f_p_plot, B_max_plot, f_fail, 'linear', 'extrap');
plot(f_fail, B_fail_max, 's', 'MarkerSize', 6, ...
    'MarkerFaceColor', 'w', 'MarkerEdgeColor', color_curve, 'LineWidth', 1.1);
text(f_fail - 0.4, B_fail_max + 0.2, ...
    sprintf('B_{max} \\approx %.2f GHz', B_fail_max), ...
    'FontName', font_en, 'FontSize', 7.5, 'Color', color_curve, ...
    'HorizontalAlignment', 'right');

% --- E. 区域文字标注 ---
text(30.8, 6.4, '失效区 (Failure Zone)', ...
    'FontName', font_cn, 'FontSize', 11, 'Color', [0.6, 0.2, 0.2], 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(30.8, 5.8, '$\xi > 1$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Color', [0.6, 0.2, 0.2], 'HorizontalAlignment', 'center');

text(26.8, 1.7, '安全区 (Safe Zone)', ...
    'FontName', font_cn, 'FontSize', 11, 'Color', [0.1, 0.5, 0.1], 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(26.8, 1.2, '$\xi < 1$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Color', [0.1, 0.5, 0.1], 'HorizontalAlignment', 'center');

%% 4. 坐标轴与标题
xlabel('截止频率 f_p (GHz)', 'Interpreter', 'tex', 'FontName', font_cn, 'FontSize', 11);
ylabel('允许最大带宽 B_{max} (GHz)', 'Interpreter', 'tex', 'FontName', font_cn, 'FontSize', 11);

% 标题（论文最终版可注释掉）
% title('色散效应工程判据约束曲线', 'FontName', font_cn, 'FontSize', 12);

xlim([x_limit_min, x_limit_max]);
ylim([0, y_limit_max]);

legend([h_pt_safe, h_pt_fail], {'典型透射诊断工况', '失效工况'}, ...
    'FontName', font_cn, 'FontSize', 9, 'Location', 'northeast', 'Box', 'on');

%% 5. 高清输出
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

print('-dpng', '-r600', fullfile(output_dir, '图3-7_色散判据_PhD版本.png'));
print('-dsvg', fullfile(output_dir, '图3-7_色散判据_PhD版本.svg'));

fprintf('✓ 博士论文标准插图已生成。\n');
