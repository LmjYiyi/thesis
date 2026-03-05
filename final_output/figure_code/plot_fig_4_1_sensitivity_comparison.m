%% plot_fig_4_1_sensitivity_comparison.m
% 论文图 4-1：电子密度与碰撞频率对群时延的差异化影响

clc; clear; close all;

%% 1. 全局物理常量
c = 2.99792458e8;
e = 1.60217663e-19;
me = 9.10938356e-31;
eps0 = 8.85418781e-12;

% 几何参数与空气背景时延
z_p1 = -231e-3; z_p2 = 954.548e-3; d = 0.15;
L_air = (z_p2 - z_p1) - d;
tau_air_base = L_air / c;

% 等离子体基准参数
fp_base = 28.98e9;
ne_base = (2*pi*fp_base)^2 * eps0 * me / e^2;

% 频率范围 (20-40 GHz)
f = linspace(20e9, 40e9, 1000);
omega = 2*pi*f;

%% 2. 创建画布 (调整为最佳长宽比)
fig = figure('Name', 'Sensitivity Comparison', 'Color', 'w', 'Position', [100, 100, 1000, 480]);
tiled = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% =============================================================
%% 子图(a): 电子密度敏感性分析
%% =============================================================
ax1 = nexttile(tiled, 1);
ne_scales = [0.6, 0.8, 1.0, 1.2, 1.4];
nu_demo = 1.5e9;
colors_ne = {'b', 'c', 'k', 'm', 'r'};
line_styles = {'--', '--', '-', '-.', '-.'};

hold(ax1, 'on'); grid(ax1, 'on');

p_handles = [];
tau_data_ne = [];
fp_list = [];

for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    [tau_g_plasma, ~] = calculate_drude_response(omega, ne_current, nu_demo, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    tau_data_ne = [tau_data_ne; tau_g_total];

    p = plot(ax1, f/1e9, tau_g_total*1e9, 'Color', colors_ne{i}, 'LineWidth', 2.0, 'LineStyle', line_styles{i});
    p_handles = [p_handles, p];

    fp_val = sqrt(ne_current * e^2 / (eps0 * me)) / (2*pi);
    fp_list = [fp_list, fp_val];
end

% [绝对防御豆腐块]：每个中文词前显式加上 \fontname{SimSun}
xlabel(ax1, '\fontname{SimSun}探测频率 \fontname{Times New Roman}\itf \rm(GHz)', 'Interpreter', 'tex', 'FontSize', 11);
ylabel(ax1, '\fontname{SimSun}群时延 \fontname{Times New Roman}\it\tau_g \rm(ns)', 'Interpreter', 'tex', 'FontSize', 11);

xlim(ax1, [20, 40]);
ylim(ax1, [3.5, 10.5]);

% 子图序号放在左上角，绝对安全
text(ax1, 0.02, 0.97, '\fontname{Times New Roman}\bf{(a)}', 'Units', 'normalized', 'Interpreter', 'tex', 'FontSize', 12);

set(ax1, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1.0, ...
    'Box', 'on', 'TickDir', 'in', 'Layer', 'top');

%% =============================================================
%% 子图(b): 碰撞频率钝感性分析
%% =============================================================
ax2 = nexttile(tiled, 2);
nu_list = [1.5e9, 3.0e9, 5.0e9, 8.0e9, 12.0e9];
colors_nu = {[0 0.6 0], [0 0.4 0.8], [0 0 0.8], [0.5 0 0.5], [0.8 0.4 0]};

h_lines = [];
tau_data_nu = [];
mag_data_nu = [];

% 左轴：时延
yyaxis(ax2, 'left'); hold(ax2, 'on'); grid(ax2, 'on');
for i = 1:length(nu_list)
    nu_current = nu_list(i);
    [tau_g_plasma, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;

    tau_data_nu = [tau_data_nu; tau_g_total];
    mag_data_nu = [mag_data_nu; mag_dB];

    ln = plot(ax2, f/1e9, tau_g_total*1e9, 'Color', colors_nu{i}, 'LineWidth', 1.5, 'LineStyle', '-');

    markers = {'o', 's', '^', 'd', 'p'};
    set(ln, 'Marker', markers{i}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
           'MarkerEdgeColor', colors_nu{i}, 'MarkerIndices', 1:100:length(f));

    h_lines = [h_lines, ln];
end

% 右轴：幅度
yyaxis(ax2, 'right'); hold(ax2, 'on');
for i = 1:length(nu_list)
    ln_mag = plot(ax2, f/1e9, mag_data_nu(i, :), 'Color', colors_nu{i}, 'LineWidth', 1.2, 'LineStyle', '--');
    set(ln_mag, 'Marker', markers{i}, 'MarkerSize', 4, 'MarkerFaceColor', 'w', ...
               'MarkerEdgeColor', colors_nu{i}, 'MarkerIndices', 50:100:length(f));
end

% 轴属性设置
yyaxis(ax2, 'left');
ylabel(ax2, '\fontname{SimSun}群时延 \fontname{Times New Roman}\it\tau_g \rm(ns)', 'Color', 'k', 'Interpreter', 'tex', 'FontSize', 11);
set(ax2, 'YColor', 'k');
ylim(ax2, [3.5, 10.5]);

yyaxis(ax2, 'right');
ylabel(ax2, '\fontname{SimSun}透射幅度 \fontname{Times New Roman}\itS_{\rm21} \rm(dB)', 'Color', 'k', 'Interpreter', 'tex', 'FontSize', 11);
set(ax2, 'YColor', 'k');
ylim(ax2, [-60, 15]);

xlabel(ax2, '\fontname{SimSun}探测频率 \fontname{Times New Roman}\itf \rm(GHz)', 'Interpreter', 'tex', 'FontSize', 11);
text(ax2, 0.02, 0.97, '\fontname{Times New Roman}\bf{(b)}', 'Units', 'normalized', 'Interpreter', 'tex', 'FontSize', 12);

xlim(ax2, [20, 40]);

set(ax2, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1.0, ...
    'Box', 'on', 'TickDir', 'in', 'Layer', 'top');

%% =============================================================
%% 导出环节 (安全无损版)
%% =============================================================
drawnow;
out_dir = fullfile(pwd, 'figures_export');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_name = 'fig_4_1_sensitivity_comparison_Final';

% 将尺寸单位设为厘米，符合论文常用宽度 14cm~16cm
set(fig, 'Units', 'centimeters', 'Position', [2, 2, 16, 7.5]);

% 导出 TIFF (分辨率600，白底)
file_tiff = fullfile(out_dir, [out_name, '.tiff']);
exportgraphics(fig, file_tiff, 'Resolution', 600, 'BackgroundColor', 'white');
fprintf('[成功导出] %s\n', file_tiff);

% 导出 EMF
file_emf = fullfile(out_dir, [out_name, '.emf']);
try
    exportgraphics(fig, file_emf, 'ContentType', 'vector', 'BackgroundColor', 'white');
    fprintf('[成功导出] %s\n', file_emf);
catch
    warning('当前平台不支持 EMF 矢量导出。');
end


%% =============================================================
%% 内部函数: Drude模型响应计算
%% =============================================================
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
