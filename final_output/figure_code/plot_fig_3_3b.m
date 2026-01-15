%% plot_fig_3_3b.m
% 论文图 3-3b:碰撞频率的"时延钝感-幅度敏感"解耦特性
% 生成日期: 2026-01-12
% 对应章节: 3.2.3 参数敏感性分析

% 【与论文文档的对应关系】
% 文档描述(第73-80行):
% - "固定电子密度 n_e = 1.04×10^19 m^-3 (对应 f_p = 29 GHz)"
% - "设置三组碰撞频率参数: ν_e = 0.1 GHz(低), 1.5 GHz(基准), 3.0 GHz(高)"
% - "双纵轴设计,左轴对应群时延(实线),右轴对应透射幅度(虚线)"
%
% 物理特征:
% 1. 时延曲线的"钝感性":三条实线几乎完全重合
% 2. 幅度衰减的"敏感性":虚线随 ν_e 变化剧烈
% 3. 定量数据:最大时延差异仅约20 ps,相对误差不足4%

clear; clc; close all;

%% 1. 全局物理常量与仿真参数(与 nue.m 完全一致)
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 仿真频段 (Ka波段扩展)
f_start = 20e9;
f_end = 40e9;
N_points = 1000;            
f = linspace(f_start, f_end, N_points);
omega = 2 * pi * f;

% 介质几何参数
d = 0.15;                   % 等离子体厚度 (m)

%% 2. 基础参数设定(与 nue.m 第77行一致)
fp_base = 29e9;             % 截止频率 (Hz),固定
wp_base = 2 * pi * fp_base;
ne_base = (wp_base^2 * eps0 * me) / e^2; % 1.04×10^19 m^-3,固定

%% 3. 碰撞频率变化:0.1, 1.5, 3.0 GHz
nu_list = [0.1e9, 1.5e9, 3.0e9]; % 低碰撞、基准、高碰撞
colors_nu = {[0 0.6 0], 'k', 'm'}; % 绿色、黑色、品红
legend_str_nu = {};

%% 4. 创建双纵轴图形
figure('Name', 'Fig 3-3b: Collision Frequency Sensitivity', ...
       'Color', 'w', ...
       'Position', [100, 100, 800, 600]);

%% 5. 绘制曲线(双Y轴)
for i = 1:length(nu_list)
    nu_current = nu_list(i);
    
    % 计算群时延和透射幅度
    [tau_g, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    
    % 左轴:群时延(实线)
    yyaxis left;
    hold on;
    plot(f/1e9, tau_g*1e9, ...
         'Color', colors_nu{i}, ...
         'LineWidth', 2.0, ...
         'LineStyle', '-');
    
    % 右轴:透射幅度(虚线)
    yyaxis right;
    hold on;
    plot(f/1e9, mag_dB, ...
         'Color', colors_nu{i}, ...
         'LineWidth', 1.5, ...
         'LineStyle', '--');
    
    legend_str_nu{end+1} = sprintf('\\nu_e = %.1f GHz', nu_current/1e9);
end

%% 6. 左轴设置(群时延)
yyaxis left;
ylabel('群时延 (ns)', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
set(gca, 'YColor', 'k');
axis tight; % 自动适应数据范围
yl_left = ylim; 
ylim([yl_left(1)*0.9, yl_left(2)*1.1]); % 微调边距

%% 7. 右轴设置(透射幅度)
yyaxis right;
ylabel('透射幅度 S_{21} (dB)', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
set(gca, 'YColor', 'r');
ylim([-35, 5]); % 幅度范围(根据nue.m的数据范围)

%% 8. 通用设置
grid on;
xlabel('探测频率 (GHz)', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
title('图 3-3b 碰撞频率的"时延钝感"与"幅度敏感"解耦特性', ...
      'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');

% 图例
legend(legend_str_nu, 'Location', 'SouthWest', 'FontSize', 12);

% 字体设置
set(gca, 'FontName', 'SimHei', 'FontSize', 12);
set(gca, 'LineWidth', 1.2);
box on;

% 添加辅助说明文本
text(22, yl_left(1)+0.1, '实线: 群时延 (左轴)', ...
     'FontSize', 10, 'Color', 'k', 'FontName', 'SimHei');
text(22, yl_left(1)+0.05, '虚线: 幅度 (右轴)', ...
     'FontSize', 10, 'Color', 'r', 'FontName', 'SimHei');


%% 10. 物理模型计算函数(与 nue.m 第127-152行一致)
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
