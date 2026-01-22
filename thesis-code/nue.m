%% LFMCW 等离子体诊断论文 - 3.2.3 仿真 (核心期刊精修版)
% 优化内容：
% 1. 修复 Fig 3-3b 图例颜色不对应问题 (使用 Handle 绑定)。
% 2. 解决文字遮挡标题问题，优化布局。
% 3. 在高衰减区增加 "加权系数 -> 0" 的物理标注，呼应反演算法。
% 4. 统一两图的物理模型 (含空气路径)。

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

% 几何参数 (CST 模型对齐)
z_p1 = -231e-3; z_p2 = 954.548e-3; d = 0.15;                   
L_air = (z_p2 - z_p1) - d;  
tau_air_base = L_air / c;   

%% 2. 基础参数
fp_base = 28.98e9; wp_base = 2 * pi * fp_base;
ne_base = (wp_base^2 * eps0 * me) / e^2; 

%% =============================================================
%% 3. 仿真 A: 电子密度 ne 的主导性分析 (Figure 3-3a)
%% =============================================================
% 保持 nu = 1.5G 以符合实际，但聚焦透射区细节
nu_demo = 1.5e9; 

figure('Name', 'Fig 3-3a: Electron Density Sensitivity', 'Color', 'w', 'Position', [100, 100, 700, 500]);

ne_scales = [0.9, 1.0, 1.1];
colors = {'b', 'k', 'r'}; 
line_styles = {'--', '-', '-.'};
legend_str = {}; 
p_handles = []; % 句柄容器

hold on; grid on; grid minor; 

% 1. 画理论截止频率辅助线
for i = 1:length(ne_scales)
    ne_val = ne_base * ne_scales(i);
    fp_val = sqrt(ne_val * e^2 / (eps0 * me)) / (2*pi);
    xline(fp_val/1e9, 'Color', [0.6 0.6 0.6], 'LineStyle', ':', 'LineWidth', 1.2);
end

% 2. 画主曲线
for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    [tau_g_plasma, ~] = calculate_drude_response(omega, ne_current, nu_demo, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    % 捕获句柄 p
    p = plot(f/1e9, tau_g_total*1e9, 'Color', colors{i}, 'LineWidth', 2.0, 'LineStyle', line_styles{i});
    p_handles = [p_handles, p];
    
    fp_current = sqrt(ne_current * e^2 / (eps0 * me)) / (2*pi);
    legend_str{end+1} = sprintf('n_e (对应 f_p = %.1f GHz)', fp_current/1e9);
end

xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('端口-端口总群时延 (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title({'图 3-3a: 电子密度对总群时延曲线的拓扑控制'}, 'FontSize', 13);

% 视野微调
xlim([27, 40]);
ylim([3.5, 8]); 

% 强制指定图例句柄，防止乱序
legend(p_handles, legend_str, 'Location', 'NorthEast', 'FontSize', 11);


%% =============================================================
%% 4. 仿真 B: 碰撞频率 nu 的解耦特性分析 (Figure 3-3b)
%% =============================================================
% 参数设定
nu_list = [1.5e9, 3e9, 5.0e9]; 
colors_nu = {'[0 0.7 0]', 'k', 'm'}; % 绿, 黑, 品红

figure('Name', 'Fig 3-3b: Collision Frequency Sensitivity', 'Color', 'w', 'Position', [850, 100, 700, 500]);

% 绘制灰色背景 (高衰减区)
x_patch = [20, 30.5, 30.5, 20];
y_patch_left = [-5, -5, 30, 30]; 
patch(x_patch, y_patch_left, [0.94 0.94 0.94], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

% 【优化】灰色区域文字说明：增加“幅度加权”暗示
text(21, 12, {'截止/高衰减区', '信噪比极低', '\bf{权重 \it{w_i} \rightarrow 0}'}, ...
    'FontSize', 11, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'left');

legend_str_nu = {};
h_lines = []; % 专门用来存左轴的线，用于生成图例

for i = 1:length(nu_list)
    nu_current = nu_list(i);
    [tau_g_plasma, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    % 左轴 (时延)
    yyaxis left; hold on;
    % 【关键】只抓取左轴的线句柄用于图例，避免和右轴混淆
    ln = plot(f/1e9, tau_g_total*1e9, 'Color', colors_nu{i}, 'LineWidth', 2, 'LineStyle', '-');
    h_lines = [h_lines, ln]; 
    
    % 右轴 (幅度)
    yyaxis right; hold on;
    plot(f/1e9, mag_dB, 'Color', colors_nu{i}, 'LineWidth', 1.5, 'LineStyle', '--');
    
    legend_str_nu{end+1} = sprintf('\\nu_e = %.1f GHz', nu_current/1e9);
end

% --- 左轴设置 ---
yyaxis left;
ylabel('端口-端口总群时延 (ns)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
set(gca, 'YColor', 'k');
xlim([20, 40]);
ylim([4.2, 14]); 

% --- 右轴设置 ---
yyaxis right;
ylabel('透射幅度 S_{21} (dB)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
set(gca, 'YColor', 'r');
ylim([-40, 5]); 

grid on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
% 标题
title({'图 3-3b: 碰撞频率的“时延钝感”与“幅度敏感”解耦特性'}, 'FontSize', 13);

% --- 辅助标注优化 ---
% 移动到右下角或不妨碍视线的地方
% 使用 text 的 normalized unit 避免硬编码坐标
text(34, -25, {'—— 实线: 群时延 (左轴)', '- - - 虚线: S21幅度 (右轴)'}, ...
     'FontSize', 10, 'Color', 'k', 'BackgroundColor', 'w', 'EdgeColor', 'k');

xline(30.5, 'k--', 'LineWidth', 1.5);
text(31.2, 0, '有效透射窗口 \rightarrow', 'FontSize', 11, 'Color', 'k', 'FontWeight', 'bold');

% 【关键】使用抓取的 h_lines 生成图例，保证颜色一一对应
legend(h_lines, legend_str_nu, 'Location', 'SouthEast', 'FontSize', 11);


%% =============================================================
%% 5. 物理内核函数
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