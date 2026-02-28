%% plot_fig_4_2_flat_valley_3d.m
% 论文图 ：参数空间中残差曲面的三维地形（平底谷可视化）
% 修正版：修复 colorbar 报错，增强视觉效果
% 对应章节：4.1.2 逆问题的病态性分析

clear; clc; close all;

%% 1. 物理常数与基本参数
c = 2.99792458e8;           
eps0 = 8.85418781e-12;      
me = 9.10938356e-31;        
e = 1.60217663e-19;         

% 真值参数
ne_true = 1.04e19;          % 电子密度真值 (m^-3)
nu_true = 1.5e9;            % 碰撞频率真值 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 测量频率范围 (Ka波段)
f_meas = linspace(26e9, 30e9, 50);
omega_meas = 2*pi*f_meas;

%% 2. 生成"观测"时延（含噪声）
sigma_noise = 0.05e-9;  % 0.05 ns 噪声

% 计算真值时延
[tau_true, ~] = calculate_group_delay_exact(omega_meas, ne_true, nu_true, d, c, eps0, me, e);
% 添加噪声
rng(42); % 固定随机种子以保证复现性
tau_meas = tau_true + sigma_noise * randn(size(tau_true));

%% 3. 构建参数空间网格
% ne 范围：±20%
ne_range = linspace(0.8*ne_true, 1.2*ne_true, 60);   
% nu 范围：0.1 GHz - 10 GHz (宽范围以展示平坦性)
nu_range = linspace(0.1e9, 10e9, 60);                 

[NE, NU] = meshgrid(ne_range, nu_range);
J_surface = zeros(size(NE));

%% 4. 计算残差曲面
fprintf('正在计算残差曲面...\n');
for i = 1:size(NE, 1)
    for j = 1:size(NE, 2)
        [tau_model, ~] = calculate_group_delay_exact(omega_meas, NE(i,j), NU(i,j), d, c, eps0, me, e);
        residual = tau_meas - tau_model;
        % 残差平方和 (Objective Function)
        J_surface(i,j) = sum(residual.^2);
    end
end

% 对数变换以便观察：log10(SSE)
% 减去最小值以便归一化视觉效果，防止数值过大
J_log = log10(J_surface); 

%% 5. 模拟优化路径（梯度下降在平底谷中震荡）
% 这是一个示意性的路径，用于展示算法在 nu 方向的无力
ne_path = [0.85*ne_true]; 
nu_path = [8e9];          

% 模拟：ne 迅速收敛，nu 随机游走/震荡
for step = 1:15
    % n_e方向：梯度很强，快速逼近真值 (指数衰减模拟收敛)
    current_ne_err = ne_true - ne_path(end);
    ne_new = ne_path(end) + 0.6 * current_ne_err; % 快速收敛
    
    % ν_e方向：梯度极弱，模拟受噪声影响的随机游走和震荡
    % 步长随迭代并未明显减小，模拟无法收敛
    drift = 1.5e9 * (-1)^step * exp(-step/10); 
    nu_new = nu_path(end) + drift;
    
    % 边界限制
    nu_new = max(0.5e9, min(9.5e9, nu_new));
    
    ne_path = [ne_path, ne_new];
    nu_path = [nu_path, nu_new];
end

% 计算路径上的 Z 值（用于绘图）
% 这里为了让路径浮在曲面上方一点点，插值获取 Z 值
J_path_vals = interp2(NE, NU, J_log, ne_path, nu_path, 'spline');

%% 6. 绘制3D曲面
figure('Position', [100, 100, 1000, 700], 'Color', 'w');

% 坐标转换：
% X轴：电子密度相对误差 (%)
% Y轴：碰撞频率 (GHz)
X_plot = (NE - ne_true) / ne_true * 100;
Y_plot = NU / 1e9;
Z_plot = J_log;

% 绘制曲面
s = surf(X_plot, Y_plot, Z_plot);
s.EdgeColor = 'none';
s.FaceAlpha = 0.9;
s.FaceLighting = 'gouraud';
hold on;

% 绘制真值点 (绿三角)
h_true = plot3(0, nu_true/1e9, min(Z_plot(:)), 'g^', ...
    'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 1.5);

% 绘制优化路径 (红线)
path_x = (ne_path - ne_true) / ne_true * 100;
path_y = nu_path / 1e9;
path_z = J_path_vals + 0.1; % 稍微抬高一点防止被面遮挡

h_path = plot3(path_x, path_y, path_z, 'r.-', ...
    'LineWidth', 2, 'MarkerSize', 15);

% 标注起点和终点
text(path_x(1), path_y(1), path_z(1)+0.5, ' 起点 (Start)', ...
    'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
text(path_x(end), path_y(end), path_z(end)+0.5, ' 震荡/不收敛', ...
    'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');

% 设置视角
view(-25, 50); % 调整视角以看清谷底的平坦
colormap(jet);

% --- 修复 colorbar 部分 ---
c = colorbar;
c.Label.String = '目标函数值 log_{10}(SSE)';
c.Label.FontSize = 11;
c.Label.FontWeight = 'bold';
% -------------------------

% 坐标轴标签
xlabel('电子密度相对误差 \delta n_e (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('碰撞频率 \nu_e (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('残差对数 log_{10}(J)', 'FontSize', 12, 'FontWeight', 'bold');

title('参数空间残差曲面：平底谷 (Flat Valley) 现象', 'FontSize', 14);

grid on; box on;
set(gca, 'FontSize', 10, 'LineWidth', 1.1);

% 添加图例
legend([h_true, h_path], {'真值点 (True Value)', '优化路径 (Optimization Path)'}, ...
    'Location', 'best');

% 插入解释性文本框
annotation('textbox', [0.15, 0.75, 0.3, 0.1], ...
    'String', {'\bf病态性特征:', ...
    '1. n_e方向: 陡峭峡谷 -> 强梯度，易收敛', ...
    '2. \nu_e方向: 平坦谷底 -> 弱梯度，无方向性'}, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', 'k');

hold off;

% 保存
export_thesis_figure(gcf, '图4-2_参数空间平底谷3D曲面', 14, 300, 'SimHei');
fprintf('图 4-2 已保存。\n');

%% 辅助函数
function [tau_g, mag_dB] = calculate_group_delay_exact(omega, ne, nu, d, c, eps0, me, e)
    % 严格计算 Drude 模型响应
    wp = sqrt(ne * e^2 / (eps0 * me));
    % 介电常数
    eps_complex = 1 - wp^2 ./ (omega.^2 + 1i.*omega.*nu);
    
    % 传播常数 k = omega/c * sqrt(eps)
    gamma = 1i * (omega/c) .* sqrt(eps_complex);
    
    % 透射系数 T = exp(-gamma * d) (忽略界面反射的简化模型，主要看色散)
    % 注：为了计算群时延，这里用相位求导
    H = exp(-gamma * d);
    
    phase = unwrap(angle(H));
    
    % 数值求导计算群时延
    % 使用中心差分或其他方法，这里简单用 diff
    d_omega = gradient(omega);
    tau_g = -gradient(phase) ./ d_omega;
    
    mag_dB = 20*log10(abs(H));
end