%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 图3-3：MATLAB理论计算 - Drude模型群时延特性（并排子图）
% (a) 电子密度敏感性（固定碰撞频率）
% (b) 碰撞频率敏感性（固定截止频率）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 全局物理常量
c = 2.99792458e8;
e = 1.60217663e-19;
me = 9.10938356e-31;
eps0 = 8.85418781e-12;

% 仿真频段
f_start = 20e9; f_end = 40e9;
N_points = 5000;
f = linspace(f_start, f_end, N_points);
omega = 2 * pi * f;

% 几何参数（与 CST 模型对齐）
z_p1 = -231e-3;
z_p2 = 954.548e-3;
d = 0.15;
L_air = (z_p2 - z_p1) - d;
tau_air_base = L_air / c;

%% 2. 基础参数
fp_base = 28.98e9;
wp_base = 2 * pi * fp_base;
ne_base = (wp_base^2 * eps0 * me) / e^2;

%% 3. 使用 tiledlayout 绘图（避免 subplot 挤压）
fig = figure('Name', 'MATLAB理论计算', 'Color', 'w', 'Position', [50, 100, 1400, 520]);
t = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>

%% 4. 子图 (a)：电子密度敏感性
ax1 = nexttile;
hold(ax1, 'on'); grid(ax1, 'on');

nu_demo = 1.5e9;
ne_scales = [0.9, 1.0, 1.1];
colors = {'b', 'k', 'r'};
line_styles = {'--', '-', '-.'};
h_left = gobjects(0);
legend_left = {};

for i = 1:length(ne_scales)
    ne_val = ne_base * ne_scales(i);
    fp_val = sqrt(ne_val * e^2 / (eps0 * me)) / (2 * pi);
    xline(ax1, fp_val / 1e9, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2);
end

for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    [tau_g_plasma, ~] = calculate_drude_response(omega, ne_current, nu_demo, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;

    h_left(end + 1) = plot(ax1, f / 1e9, tau_g_total * 1e9, ...
        'Color', colors{i}, 'LineWidth', 2.0, 'LineStyle', line_styles{i}); %#ok<SAGROW>

    fp_current = sqrt(ne_current * e^2 / (eps0 * me)) / (2 * pi);
    legend_left{end + 1} = sprintf('\\itf_p \\rm= %.1f GHz', fp_current / 1e9); %#ok<SAGROW>
end

xlabel(ax1, '探测频率 \fontname{Times New Roman}(GHz)', 'FontSize', 11, 'Interpreter', 'tex');
ylabel(ax1, '群时延 \fontname{Times New Roman}(ns)', 'FontSize', 11, 'Interpreter', 'tex');
xlim(ax1, [20, 40]);
ylim(ax1, [3.5, 14]);
legend(ax1, h_left, legend_left, 'Location', 'northeast', 'FontSize', 9, 'Interpreter', 'tex');
text(ax1, 0.03, 0.95, '\textbf{(a)}', 'Units', 'normalized', ...
    'Interpreter', 'latex', 'FontSize', 12, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
set(ax1, 'FontSize', 10);

%% 5. 子图 (b)：碰撞频率敏感性（双 Y 轴）
ax2 = nexttile;
hold(ax2, 'on'); grid(ax2, 'on');

nu_list = [1.5e9, 3.0e9, 5.0e9];
colors_nu = {[0 0.6 0], [0 0 0.8], [0.8 0 0.6]};
h_right_legend = gobjects(0);
legend_right = {};

for i = 1:length(nu_list)
    nu_current = nu_list(i);
    [tau_g_plasma, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;

    yyaxis(ax2, 'left');
    h_right_legend(end + 1) = plot(ax2, f / 1e9, tau_g_total * 1e9, ...
        'Color', colors_nu{i}, 'LineWidth', 2.0, 'LineStyle', '-'); %#ok<SAGROW>

    yyaxis(ax2, 'right');
    plot(ax2, f / 1e9, mag_dB, ...
        'Color', colors_nu{i}, 'LineWidth', 1.5, 'LineStyle', '--');

    legend_right{end + 1} = sprintf('\\it\\nu_e \\rm= %.1f GHz', nu_current / 1e9); %#ok<SAGROW>
end

yyaxis(ax2, 'left');
ylabel(ax2, '群时延 \fontname{Times New Roman}(ns)', 'FontSize', 11, 'Color', 'k', 'Interpreter', 'tex');
ax2.YColor = 'k';
xlim(ax2, [20, 40]);
ylim(ax2, [3.5, 14]);

yyaxis(ax2, 'right');
ylabel(ax2, '透射幅度 \fontname{Times New Roman}\itS_{\rm21}\rm (dB)', 'FontSize', 11, ...
    'Color', [0.6350 0.0780 0.1840], 'Interpreter', 'tex');
ax2.YColor = [0.6350 0.0780 0.1840];
ylim(ax2, [-40, 15]);

xlabel(ax2, '探测频率 \fontname{Times New Roman}(GHz)', 'FontSize', 11, 'Interpreter', 'tex');
xline(ax2, fp_base / 1e9, 'k--', 'LineWidth', 1.2);
yyaxis(ax2, 'left');
legend(ax2, h_right_legend, legend_right, 'Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
text(ax2, 0.03, 0.95, '\textbf{(b)}', 'Units', 'normalized', ...
    'Interpreter', 'latex', 'FontSize', 12, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
set(ax2, 'FontSize', 10);

%% 6. 导出
export_thesis_figure(fig, '图3-3_MATLAB理论计算_完美版', 14, 600, 'SimSun');

%% 7. 辅助函数：Drude 模型响应计算
function [tau_g, mag_dB] = calculate_drude_response(omega, ne, nu, d, c, eps0, me, e)
    wp = sqrt(ne * e^2 / (eps0 * me));
    term_denom = omega.^2 + nu^2;
    eps_real = 1 - (wp^2) ./ term_denom;
    eps_imag = -(nu ./ omega) .* (wp^2 ./ term_denom);
    eps_complex = eps_real + 1i * eps_imag;
    k0 = omega ./ c;
    gamma = 1i .* k0 .* sqrt(eps_complex);
    H = exp(-gamma * d);
    mag_dB = 20 * log10(abs(H));
    phi = unwrap(angle(H));
    d_phi = diff(phi);
    d_omega = diff(omega);
    tau_g_raw = -d_phi ./ d_omega;
    tau_g = [tau_g_raw, tau_g_raw(end)];
end
