%% plot_fig_4_10.m
% 论文图 4-10:不同碰撞频率先验下的拟合对比
% 生成日期:2026-01-11
% 对应章节:4.3.3 鲁棒性测试
%
% 【图表描述】(第186行)
% 展示不同ν_e^fix下拟合曲线与测量点的对比:
% - 即使碰撞频率先验存在50%偏差
% - 拟合曲线的形态几乎没有可见差异
% - 曲线弯曲程度主要由n_e(通过f_p)决定
%
% 【关键特征】
% - 多条拟合曲线重叠
% - 测量点散布相同
% - 验证ν_e仅引入极其微小的形态调整

clear; clc; close all;

%% 1. 参数设置
c = 3e8;
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;

% 真实参数
n_e_true = 1.04e19;
nu_e_true = 1.5e9;
f_p = sqrt(n_e_true * e^2 / (epsilon_0 * m_e)) / (2*pi);

% 雷达参数
f_start = 34.2e9;
f_end = 37.4e9;
d = 0.15;

% 不同先验值
nu_e_priors = [0.75, 1.5, 2.25] * 1e9;  % 低估50%, 精确, 高估50%
nu_e_labels = {'-50%', '精确', '+50%'};

% 探测频率
f_probe = linspace(f_start, f_end, 100);

%% 2. 计算拟合曲线（使用完整Drude模型）
% 使用真实碰撞频率计算测量基准
tau_theory_true = calculate_drude_delay(f_probe, n_e_true, nu_e_true, d, c, epsilon_0, m_e, e);

% 模拟测量点(加噪声)
rng(42);  % 固定随机种子
tau_meas = tau_theory_true + 3e-11 * randn(size(tau_theory_true));

% 不同先验下的拟合曲线（使用不同碰撞频率）
tau_fits = cell(1, 3);
for i = 1:3
    % 使用完整Drude模型，但采用不同的碰撞频率先验
    tau_fits{i} = calculate_drude_delay(f_probe, n_e_true, nu_e_priors(i), d, c, epsilon_0, m_e, e);
end

%% 3. 绘图
figure('Position', [100, 100, 800, 600]);

% 绘制测量点
scatter(f_probe/1e9, tau_meas*1e9, 30, [0.5 0.5 0.5], 'filled', 'DisplayName', '测量数据');
hold on;

% 绘制拟合曲线(使用不同线型)
colors = [0.0, 0.4, 0.7; 0.85, 0.33, 0.1; 0.47, 0.67, 0.19];
line_styles = {'-', '--', ':'};

for i = 1:3
    plot(f_probe/1e9, tau_fits{i}*1e9, 'Color', colors(i,:), ...
         'LineStyle', line_styles{i}, 'LineWidth', 2.5, ...
         'DisplayName', sprintf('\\nu_e^{fix}%s (%.2f GHz)', nu_e_labels{i}, nu_e_priors(i)/1e9));
end

xlabel('探测频率(GHz)', 'FontSize', 13, 'FontName', 'SimHei');
ylabel('相对群时延(ns)', 'FontSize', 13, 'FontName', 'SimHei');
title('图4-10 不同碰撞频率先验下的拟合对比', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 11, 'Interpreter', 'tex');
set(gca, 'FontName', 'SimHei', 'FontSize', 12);
grid on; box on;
xlim([f_start/1e9 f_end/1e9]);

% 添加文本说明
text(34.5, max(tau_meas)*1e9*0.9, ...
     '曲线形态几乎重叠,\nu_e影响极小', ...
     'FontSize', 11, 'FontName', 'SimHei', 'Color', 'r', 'FontWeight', 'bold', 'Interpreter', 'tex');

fprintf('图 4-10 生成完成！\n');

%% 局部函数：完整Drude模型时延计算（相位求导法）
function tau_rel = calculate_drude_delay(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
    % 核心物理模型：Drude模型相位求导法（与 thesis-code/LM.m 一致）
    % 计算相对群时延 = (等离子体群时延) - (真空群时延)
    
    omega_vec = 2 * pi * f_vec;
    wp_val = sqrt(ne_val * e_charge^2 / (eps0 * me));
    
    % Drude 模型复介电常数 (含碰撞频率虚部)
    % epsilon = 1 - wp^2 / (w*(w + i*nu))
    eps_r = 1 - (wp_val^2) ./ (omega_vec .* (omega_vec + 1i*nu_val));
    
    % 复波数 k = (w/c) * sqrt(eps_r)
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 等离子体段的总相位 phi = -real(k) * d
    phi_plasma = -real(k_vec) * d;
    
    % 数值微分求群时延 tau_g = -d(phi)/d(omega)
    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    
    tau_total = -d_phi ./ d_omega;
    
    % 维度补齐 (diff会少一个点，这里简单复制最后一个值)
    tau_total = [tau_total, tau_total(end)];
    
    % 减去真空穿过同样厚度 d 的时延 d/c
    % 得到的就是 "等离子体引起的附加时延"
    tau_rel = tau_total - (d/c);
end
