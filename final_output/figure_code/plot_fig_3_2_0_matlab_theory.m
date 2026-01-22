%% 图3-3: MATLAB理论计算 - Drude模型群时延特性 (左右子图)
% (a) 电子密度敏感性分析 (固定碰撞频率)
% (b) 碰撞频率敏感性分析 (固定截止频率)
% 对应章节: 3.2.2 - 用于与CST仿真对比,展示理想无多径情况下的光滑曲线

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

% 几何参数 (与CST模型对齐)
z_p1 = -231e-3; z_p2 = 954.548e-3; d = 0.15;                   
L_air = (z_p2 - z_p1) - d;  
tau_air_base = L_air / c;   

%% 2. 基础参数
fp_base = 28.98e9; wp_base = 2 * pi * fp_base;
ne_base = (wp_base^2 * eps0 * me) / e^2; 

%% 3. 创建左右子图
figure('Name', '图3-3: MATLAB理论计算', 'Color', 'w', 'Position', [50, 100, 1400, 500]);

%% =============================================================
%% 子图(a): 电子密度敏感性分析
%% =============================================================
subplot(1, 2, 1);

nu_demo = 1.5e9; 
ne_scales = [0.9, 1.0, 1.1];
colors = {'b', 'k', 'r'}; 
line_styles = {'--', '-', '-.'};
legend_str = {}; 
p_handles = [];

hold on; grid on;

% 画理论截止频率辅助线
for i = 1:length(ne_scales)
    ne_val = ne_base * ne_scales(i);
    fp_val = sqrt(ne_val * e^2 / (eps0 * me)) / (2*pi);
    xline(fp_val/1e9, 'Color', [0.6 0.6 0.6], 'LineStyle', ':', 'LineWidth', 1.2);
end

% 画主曲线
for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    [tau_g_plasma, ~] = calculate_drude_response(omega, ne_current, nu_demo, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    p = plot(f/1e9, tau_g_total*1e9, 'Color', colors{i}, 'LineWidth', 2.0, 'LineStyle', line_styles{i});
    p_handles = [p_handles, p];
    
    fp_current = sqrt(ne_current * e^2 / (eps0 * me)) / (2*pi);
    legend_str{end+1} = sprintf('f_p = %.1f GHz', fp_current/1e9);
end

xlabel('探测频率 (GHz)', 'FontSize', 11);
ylabel('群时延 (ns)', 'FontSize', 11);
title('(a) 电子密度敏感性 (固定 \nu_e = 1.5 GHz)', 'FontSize', 12);

xlim([20, 40]);
ylim([3.5, 12]); 

legend(p_handles, legend_str, 'Location', 'NorthEast', 'FontSize', 10);
set(gca, 'FontSize', 10);

%% =============================================================
%% 子图(b): 碰撞频率敏感性分析
%% =============================================================
subplot(1, 2, 2);

nu_list = [1.5e9, 3e9, 5.0e9]; 
colors_nu = {[0 0.6 0], [0 0 0.8], [0.8 0 0.6]}; % 绿, 蓝, 品红

hold on; grid on;

legend_str_nu = {};
h_lines = [];

for i = 1:length(nu_list)
    nu_current = nu_list(i);
    [tau_g_plasma, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    % 左轴 (时延)
    yyaxis left; hold on;
    ln = plot(f/1e9, tau_g_total*1e9, 'Color', colors_nu{i}, 'LineWidth', 2, 'LineStyle', '-');
    h_lines = [h_lines, ln]; 
    
    % 右轴 (幅度)
    yyaxis right; hold on;
    plot(f/1e9, mag_dB, 'Color', colors_nu{i}, 'LineWidth', 1.2, 'LineStyle', '--');
    
    legend_str_nu{end+1} = sprintf('\\nu_e = %.1f GHz', nu_current/1e9);
end

% 左轴设置
yyaxis left;
ylabel('群时延 (ns)', 'FontSize', 11, 'Color', 'k');
set(gca, 'YColor', 'k');
xlim([20, 40]);
ylim([3.5, 12]); 

% 右轴设置
yyaxis right;
ylabel('透射幅度 S_{21} (dB)', 'FontSize', 11, 'Color', [0.6 0 0]);
set(gca, 'YColor', [0.6 0 0]);
ylim([-40, 5]); 

xlabel('探测频率 (GHz)', 'FontSize', 11);
title('(b) 碰撞频率敏感性 (固定 f_p = 29.0 GHz)', 'FontSize', 12);

% 截止频率线
xline(fp_base/1e9, 'k--', 'LineWidth', 1.2);

legend(h_lines, legend_str_nu, 'Location', 'NorthEast', 'FontSize', 10);
set(gca, 'FontSize', 10);

% 总标题
sgtitle('图3-3 Drude模型理论群时延曲线 (MATLAB计算)', 'FontSize', 13, 'FontWeight', 'bold');

%% =============================================================
%% 辅助函数: Drude模型响应计算
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
