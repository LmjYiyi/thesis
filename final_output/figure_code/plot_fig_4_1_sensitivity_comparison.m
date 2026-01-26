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
    
    % 添加稀疏标记以区分曲线并避免数据点过密导致的线条过粗
    markers = {'o', 's', '^', 'd', 'p'};
    set(ln, 'Marker', markers{i}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
           'MarkerIndices', 1:100:length(f));
           
    h_lines = [h_lines, ln]; 
    
    legend_str_nu{end+1} = sprintf('\\nu_e = %.1f GHz', nu_current/1e9);
end

% 2. 再画右轴：所有幅度曲线
yyaxis right; hold on;
for i = 1:length(nu_list)
    ln_mag = plot(f/1e9, mag_data_nu(i, :), 'Color', colors_nu{i}, 'LineWidth', 1.2, 'LineStyle', '--');
    % 为幅度曲线也添加稀疏标记，起始位置错开 50 个点以避免与时延标记重叠
    set(ln_mag, 'Marker', markers{i}, 'MarkerSize', 4, 'MarkerFaceColor', 'w', ...
               'MarkerIndices', 50:100:length(f));
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
sgtitle('电子密度与碰撞频率对群时延的差异化影响', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图像
print('-dpng', '-r300', '../../final_output/figures/图4-1_电子密度与碰撞频率敏感性对比.png');
print('-dsvg', '../../final_output/figures/图4-1_电子密度与碰撞频率敏感性对比.svg');

%% =============================================================
%% 计算并打印最大偏移量 (基于统一的相对避让带宽)
%% =============================================================
fprintf('--------------------------------------------------------\n');
fprintf('敏感性分析统计结果 (Passband Analysis):\n');

% 1. 设定基准比例 (统一设定为 5% 的频率避让裕量)
ratio_offset = 0.05; 
fprintf('统计基准设定: 统一采用截止频率之上 %.2f%% 的频率点作为统计起点\n', ratio_offset * 100);

% 2. 电子密度图统计 (左图)
% 左图每一条线的截止频率不同，最坏情况是 max(fp_list)
% 应用相同的相对避让比例
max_fp_ne = max(fp_list);
f_limit_ne = max_fp_ne * (1 + ratio_offset);
valid_idx_ne = f > f_limit_ne;

if any(valid_idx_ne)
    tau_subset_ne = tau_data_ne(:, valid_idx_ne);
    diff_ne = max(tau_subset_ne) - min(tau_subset_ne); % 列向量：每个频率点上不同曲线的最大差值
    [max_offset_ne, idx_max_ne] = max(diff_ne);
    max_offset_ne_ns = max_offset_ne * 1e9; 
    
    % 对应的频率点
    f_idx_sub = find(valid_idx_ne);
    f_at_max_ne = f(f_idx_sub(idx_max_ne));
    
    % 计算相对变化率
    avg_tau_at_max_ne = mean(tau_subset_ne(:, idx_max_ne));
    percentage_variation_ne = (max_offset_ne / avg_tau_at_max_ne) * 100;
    
    fprintf('1. 电子密度 (n_e 变化 ±40%%):\n');
    fprintf('   统计起始频率 = %.2f GHz (f_p_max + %.2f%%)\n', f_limit_ne/1e9, ratio_offset*100);
    fprintf('   最大群时延偏移量 = %.4f ns (at %.2f GHz)\n', max_offset_ne_ns, f_at_max_ne/1e9);
    fprintf('   相对变化率 = %.2f%%\n', percentage_variation_ne);
else
    fprintf('1. 电子密度: 有效频段外 (f_limit > 40GHz)\n');
end

% 3. 碰撞频率图统计 (右图)
% 截止频率固定为 fp_base
f_limit_nu = fp_base * (1 + ratio_offset); 
valid_idx_nu = f > f_limit_nu;

if any(valid_idx_nu)
    tau_subset_nu = tau_data_nu(:, valid_idx_nu);
    
    % 计算该频段内，由于 nu 变化导致的最大时延差
    tau_spread_per_freq = max(tau_subset_nu) - min(tau_subset_nu);
    [max_offset_nu, idx_max_nu] = max(tau_spread_per_freq);
    
    % 对应的基准时延
    avg_tau_at_max = mean(tau_subset_nu(:, idx_max_nu));
    percentage_variation = (max_offset_nu / avg_tau_at_max) * 100;
    max_offset_nu_ns = max_offset_nu * 1e9;
    
    % 计算 nu 参数本身的变化幅度
    nu_min = min(nu_list);
    nu_max = max(nu_list);
    nu_param_change = (nu_max - nu_min) / nu_min * 100;
    
    % 对应的频率点
    f_idx_sub_nu = find(valid_idx_nu);
    f_at_max_nu = f(f_idx_sub_nu(idx_max_nu));
    
    fprintf('2. 碰撞频率 (总体: %.1f-%.1f GHz, 参数变化 +%.0f%%):\n', nu_min/1e9, nu_max/1e9, nu_param_change);
    fprintf('   统计起始频率 = %.2f GHz (f_p + %.2f%%)\n', f_limit_nu/1e9, ratio_offset*100);
    fprintf('   最大群时延偏移量 = %.4f ns (at %.2f GHz)\n', max_offset_nu_ns, f_at_max_nu/1e9);
    fprintf('   时延相对变化率 = %.2f%%\n', percentage_variation);
    
    % --- [新增] 特别统计 nu = 3.0 GHz 的情况 ---
    idx_base = find(nu_list == 1.5e9, 1);
    idx_target = find(nu_list == 3.0e9, 1);
    
    if ~isempty(idx_base) && ~isempty(idx_target)
        tau_base_row = tau_data_nu(idx_base, valid_idx_nu);
        tau_target_row = tau_data_nu(idx_target, valid_idx_nu);
        
        diff_3g = abs(tau_target_row - tau_base_row);
        [max_diff_3g, idx_max_3g] = max(diff_3g);
        
        avg_tau_3g = tau_base_row(idx_max_3g); 
        per_var_3g = (max_diff_3g / avg_tau_3g) * 100;
        
        fprintf('   [特别关注] 当 nu_e 仅从 1.5 增至 3.0 GHz (参数变化 +100%%) 时:\n');
        fprintf('   最大群时延偏移量 = %.4f ns\n', max_diff_3g * 1e9);
        fprintf('   时延相对变化率 = %.4f%%\n', per_var_3g);
    end

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
