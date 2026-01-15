%% plot_fig_3_3a.m
% 论文图 3-3a:电子密度对群时延曲线的拓扑控制
% 生成日期: 2026-01-12
% 对应章节: 3.2.3 参数敏感性分析

% 【与论文文档的对应关系】
% 文档描述(第53-65行):
% - "固定碰撞频率 ν_e = 1.5 GHz"
% - "在基准电子密度 n_e_base = 1.04×10^19 m^-3 附近设置三组参数:0.9倍、1.0倍、1.1倍"
% - "对应截止频率分别为 28.07 GHz、29.00 GHz、29.92 GHz"
% - "探测频段覆盖 Ka 波段 20-40 GHz"
% 
% 物理特征:
% 1. 曲线位移的单向性(整体平移)
% 2. 发散速率的差异化(低频段斜率差异显著)
% 3. 非线性度因子 η 的直接体现(曲线弯曲程度)

clear; clc; close all;

%% 1. 全局物理常量与仿真参数(与 nue.m 完全一致)
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 仿真频段 (Ka波段 34-38 GHz,扩展到20-40 GHz以观察全局特征)
f_start = 20e9;
f_end = 40e9;
N_points = 1000;            
f = linspace(f_start, f_end, N_points);
omega = 2 * pi * f;

% 介质几何参数
d = 0.15;                   % 等离子体厚度 (m)

%% 2. 基础参数设定(与 nue.m 第26-29行一致)
fp_base = 29e9;             % 基准截止频率 (Hz)
wp_base = 2 * pi * fp_base;
ne_base = (wp_base^2 * eps0 * me) / e^2; % 1.04×10^19 m^-3
nu_base = 1.5e9;            % 碰撞频率 (Hz),固定

%% 3. 电子密度变化:0.9倍、1.0倍、1.1倍基准值
ne_scales = [0.9, 1.0, 1.1];
colors = {'b', 'k', 'r'};   % 蓝色(低)、黑色(基准)、红色(高)
line_styles = {'--', '-', '-.'}; % 虚线、实线、点划线
legend_str = {};

%% 4. 创建图形
figure('Name', 'Fig 3-3a: Electron Density Sensitivity', ...
       'Color', 'w', ...
       'Position', [100, 100, 800, 600]);

hold on; grid on;

%% 5. 绘制三条曲线
for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    
    % 计算群时延(调用Drude模型函数)
    [tau_g, ~] = calculate_drude_response(omega, ne_current, nu_base, d, c, eps0, me, e);
    
    % 绘图 (频率单位 GHz, 时延单位 ns)
    plot(f/1e9, tau_g*1e9, ...
         'Color', colors{i}, ...
         'LineWidth', 2.0, ...
         'LineStyle', line_styles{i});
    
    % 计算对应的截止频率
    fp_current = sqrt(ne_current * e^2 / (eps0 * me)) / (2*pi);
    legend_str{end+1} = sprintf('n_e (f_p = %.2f GHz)', fp_current/1e9);
end

%% 6. 图表标注与格式设置
xlabel('探测频率 (GHz)', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
ylabel('群时延 (ns)', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
title('图 3-3a 电子密度对群时延曲线的拓扑控制', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');

% 自动适应数据范围
axis tight;
yl = ylim; ylim([yl(1)*0.95, yl(2)*1.05]); % 稍微留点边距

legend(legend_str, 'Location', 'Best', 'FontSize', 12);

% 设置图表样式
set(gca, 'FontName', 'SimHei', 'FontSize', 12);
set(gca, 'LineWidth', 1.2);
box on;


%% 8. 物理模型计算函数(与 nue.m 第127-152行一致)
function [tau_g, mag_dB] = calculate_drude_response(omega, ne, nu, d, c, eps0, me, e)
    % 计算等离子体频率
    wp = sqrt(ne * e^2 / (eps0 * me));
    
    % 计算复介电常数
    term_denom = omega.^2 + nu^2;
    eps_real = 1 - (wp^2) ./ term_denom;
    eps_imag = -(nu ./ omega) .* (wp^2 ./ term_denom); 
    eps_complex = eps_real + 1i * eps_imag;
    
    % 计算复传播常数
    k0 = omega ./ c;
    gamma = 1i .* k0 .* sqrt(eps_complex);
    
    % 传递函数
    H = exp(-gamma * d);
    
    % 幅度 (dB)
    mag_dB = 20 * log10(abs(H));
    
    % 群时延 (数值微分)
    phi = unwrap(angle(H));
    d_phi = diff(phi);
    d_omega = diff(omega);
    tau_g_raw = -d_phi ./ d_omega;
    tau_g = [tau_g_raw, tau_g_raw(end)];  % 补齐最后一个点
end
