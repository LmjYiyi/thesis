%% plot_fig_4_1_sensitivity_comparison.m
% 论文图 4-1：电子密度与碰撞频率对群时延的差异化影响
% 对应章节：4.1.1 碰撞频率的二阶微扰特性与时延不敏感机理

clear; clc; close all;

%% 1. 全局物理常量 (与 plot_fig_3_2_0 一致)
c = 2.99792458e8;           
e = 1.60217663e-19;         
me = 9.10938356e-31;        
eps0 = 8.85418781e-12;      

% 几何参数与空气背景时延 (与第3章理论计算对齐)
z_p1 = -231e-3; z_p2 = 954.548e-3; d = 0.15;                   
L_air = (z_p2 - z_p1) - d;  
tau_air_base = L_air / c;   

% 等离子体基准参数
fp_base = 28.98e9;          
ne_base = (2*pi*fp_base)^2 * eps0 * me / e^2;  

% 频率范围 (20-40 GHz)
f = linspace(20e9, 40e9, 1000);  
omega = 2*pi*f;

%% 2. 创建画布
figure('Name', '图4-1: 敏感性对比分析', 'Color', 'w', 'Position', [100, 100, 1200, 500]);

%% =============================================================
%% 子图(a): 电子密度敏感性分析 (扩大至 40% 变化, 5条曲线)
%% =============================================================
subplot(1, 2, 1);
ne_scales = [0.6, 0.8, 1.0, 1.2, 1.4];  % 变化 ±40%
nu_demo = 1.5e9;              
colors_ne = {'b', 'c', 'k', 'm', 'r'}; % 蓝, 青, 黑, 品红, 红
line_styles = {'--', '--', '-', '-.', '-.'};

hold on; grid on;

% 绘制主曲线并存储数据用于计算偏移
p_handles = [];
legend_str = {};
tau_data_ne = []; % 存储所有曲线的时延数据
fp_list = [];     % 存储所有截止频率

for i = 1:length(ne_scales)
    ne_current = ne_base * ne_scales(i);
    [tau_g_plasma, ~] = calculate_drude_response(omega, ne_current, nu_demo, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    % 存储数据 (行向量)
    tau_data_ne = [tau_data_ne; tau_g_total];
    
    p = plot(f/1e9, tau_g_total*1e9, 'Color', colors_ne{i}, 'LineWidth', 2.0, 'LineStyle', line_styles{i});
    p_handles = [p_handles, p];
    
    fp_val = sqrt(ne_current * e^2 / (eps0 * me)) / (2*pi);
    fp_list = [fp_list, fp_val];
    legend_str{end+1} = sprintf('f_p = %.1f GHz', fp_val/1e9);
end

xlabel('探测频率 (GHz)', 'FontSize', 11);
ylabel('群时延 (ns)', 'FontSize', 11);
title('(a) 电子密度敏感性 (n_e 变化 ±40%)', 'FontSize', 12);

xlim([20, 40]);
ylim([3.5, 9]); 
legend(p_handles, legend_str, 'Location', 'NorthEast', 'FontSize', 9);
set(gca, 'FontSize', 10, 'LineWidth', 1.1);

%% =============================================================
%% 子图(b): 碰撞频率钝感性分析 (5条曲线, 增加至 12 GHz)
%% =============================================================
subplot(1, 2, 2);
nu_list = [1.5e9, 3.0e9, 5.0e9, 8.0e9, 12.0e9];  % 5条曲线
colors_nu = {[0 0.6 0], [0 0.4 0.8], [0 0 0.8], [0.5 0 0.5], [0.8 0.4 0]}; % 渐变色系

h_lines = [];
legend_str_nu = {};
tau_data_nu = []; % 存储所有曲线的时延数据
mag_data_nu = []; % 存储幅度数据

% 1. 先画左轴：所有时延曲线
yyaxis left; hold on; grid on;
for i = 1:length(nu_list)
    nu_current = nu_list(i);
    [tau_g_plasma, mag_dB] = calculate_drude_response(omega, ne_base, nu_current, d, c, eps0, me, e);
    tau_g_total = tau_g_plasma + tau_air_base;
    
    % 存储数据
    tau_data_nu = [tau_data_nu; tau_g_total];
    mag_data_nu = [mag_data_nu; mag_dB]; % 行向量存储
    
    ln = plot(f/1e9, tau_g_total*1e9, 'Color', colors_nu{i}, 'LineWidth', 1.5, 'LineStyle', '-');
    h_lines = [h_lines, ln]; 
    
    legend_str_nu{end+1} = sprintf('\\nu_e = %.1f GHz', nu_current/1e9);
end

% 2. 再画右轴：所有幅度曲线
yyaxis right; hold on;
for i = 1:length(nu_list)
    plot(f/1e9, mag_data_nu(i, :), 'Color', colors_nu{i}, 'LineWidth', 1.2, 'LineStyle', '--');
end

% 左轴设置
yyaxis left;
ylabel('群时延 (ns)', 'FontSize', 11, 'Color', 'k');
set(gca, 'YColor', 'k');
ylim([3.5, 9]); 

% 右轴设置
yyaxis right;
ylabel('透射幅度 S_{21} (dB)', 'FontSize', 11, 'Color', [0.6 0 0]);
set(gca, 'YColor', [0.6 0 0]);
ylim([-60, 5]); 

xlabel('探测频率 (GHz)', 'FontSize', 11);
title('(b) 碰撞频率钝感性 (\nu_e 至 12 GHz)', 'FontSize', 12);

xlim([20, 40]);
legend(h_lines, legend_str_nu, 'Location', 'SouthWest', 'FontSize', 9);
set(gca, 'FontSize', 10, 'LineWidth', 1.1);

% 总标题
sgtitle('图 4-1 电子密度与碰撞频率对群时延的差异化影响', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图像
print('-dpng', '-r300', '../../final_output/figures/图4-1_电子密度与碰撞频率敏感性对比.png');
print('-dsvg', '../../final_output/figures/图4-1_电子密度与碰撞频率敏感性对比.svg');

%% =============================================================
%% 计算并打印最大偏移量 (仅计算最大截止频率之后的区域)
%% =============================================================
fprintf('--------------------------------------------------------\n');
fprintf('敏感性分析统计结果 (Passband Analysis):\n');

% 确定全局最大截止频率 (以最坏情况为准，保证都在传播区)
% 对于(a)图，最大截止频率是 max(fp_list)
% 对于(b)图，截止频率固定为 fp_base
% 为了公平对比，我们对两组数据分别使用其物理上的"全通频段"进行统计
% (a) 的全通频段起始点: max(fp_list)
% (b) 的全通频段起始点: fp_base (或者为了严格对比，也可以用 max(fp_list)，但此处按各自物理有效区计算更合理)

% 逻辑：用户要求"计算最大截止频率之后的频率"。
% 这里理解为：每张图各自的有效传播区。

% 1. 电子密度图统计
max_fp_ne = max(fp_list);
valid_idx_ne = f > (max_fp_ne + 0.1e9); % 留一点裕量避开极点震荡
if any(valid_idx_ne)
    tau_subset_ne = tau_data_ne(:, valid_idx_ne);
    diff_ne = max(tau_subset_ne) - min(tau_subset_ne);
    max_offset_ne = max(diff_ne) * 1e9; 
    fprintf('1. 电子密度 (±40%%, f > %.2f GHz):\n   最大群时延偏移量 = %.4f ns\n', max_fp_ne/1e9, max_offset_ne);
elseif ~isempty(fp_list) && max_fp_ne > max(f)
    fprintf('1. 电子密度: 最大截止频率 (%.2f GHz) 超出绘图频率范围 (%.2f GHz)\n', max_fp_ne/1e9, max(f)/1e9);
else
    fprintf('1. 电子密度: 有效频段外 (f_max < 40GHz)\n');
end

% 2. 碰撞频率图统计
% 碰撞频率不改变截止频率，所以截止频率一直是 fp_base
max_fp_nu = fp_base;
% 用户指定：碰撞频率的统计区间在31Ghz以上
f_limit_low = 31e9; 
valid_idx_nu = f > f_limit_low;

if any(valid_idx_nu)
    tau_subset_nu = tau_data_nu(:, valid_idx_nu);
    
    % 计算该频段内，由于 nu 变化导致的最大时延差
    % 对每一列（每个频率点），求 max - min (即不同 nu 曲线在该频率点的发散程度)
    tau_spread_per_freq = max(tau_subset_nu) - min(tau_subset_nu);
    [max_offset_nu, idx_max] = max(tau_spread_per_freq);
    
    % 对应的基准时延（用作分母计算百分比，取平均值更稳健）
    avg_tau_at_max = mean(tau_subset_nu(:, idx_max));
    
    percentage_variation = (max_offset_nu / avg_tau_at_max) * 100;
    max_offset_ns = max_offset_nu * 1e9;
    
    fprintf('2. 碰撞频率 (1.5-12 GHz, 统计区间 > %.2f GHz):\n', f_limit_low/1e9);
    fprintf('   最大群时延偏移量 = %.4f ns\n', max_offset_ns);
    fprintf('   相对变化率 = %.2f%%\n', percentage_variation);
else
    fprintf('2. 碰撞频率: 有效频段外\n');
end

fprintf('--------------------------------------------------------\n');


%% =============================================================
%% 辅助函数: Drude模型响应计算 (核心物理逻辑)
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
