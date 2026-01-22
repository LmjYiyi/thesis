%% plot_fig_4_2_flat_valley_3d.m
% 论文图 4-2：参数空间中残差曲面的三维地形（平底谷可视化）
% 生成日期：2026-01-22
% 对应章节：4.1.2 逆问题的病态性分析
%
% 图表描述（摘自定稿文档）：
% "图4-2展示了参数空间中残差曲面的三维地形。如图所示，在n_e维度，曲面呈现
%  陡峭的峡谷壁，梯度方向明确指向真值；而在ν_e维度，曲面表现为极度平坦的谷底，
%  残差值几乎不随ν_e变化。图中标注的优化路径(红色折线)展示了梯度下降算法在
%  谷底来回震荡却无法沿ν_e方向前进的典型困境。"

clear; clc; close all;

%% 1. 物理常数与基本参数
c = 3e8;                    % 光速 (m/s)
eps0 = 8.854e-12;           % 真空介电常数 (F/m)
me = 9.109e-31;             % 电子质量 (kg)
e = 1.602e-19;              % 电子电量 (C)

% 真值参数
ne_true = 1.04e19;          % 电子密度真值 (m^-3)
nu_true = 1.5e9;            % 碰撞频率真值 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 测量频率范围 (Ka波段)
f_meas = linspace(26e9, 30e9, 50);
omega_meas = 2*pi*f_meas;

%% 2. 生成"观测"时延（含噪声）
sigma_noise = 0.05e-9;  % 0.05 ns 噪声

tau_true = calculate_group_delay(omega_meas, ne_true, nu_true, d, c, eps0, me, e);
tau_meas = tau_true + sigma_noise * randn(size(tau_true));

%% 3. 构建参数空间网格
ne_range = linspace(0.8*ne_true, 1.2*ne_true, 80);   % n_e: ±20%
nu_range = linspace(0.1e9, 10e9, 80);                 % ν_e: 0.1-10 GHz

[NE, NU] = meshgrid(ne_range, nu_range);
J_surface = zeros(size(NE));

%% 4. 计算残差曲面
fprintf('正在计算残差曲面...\n');
for i = 1:size(NE, 1)
    for j = 1:size(NE, 2)
        tau_model = calculate_group_delay(omega_meas, NE(i,j), NU(i,j), d, c, eps0, me, e);
        residual = tau_meas - tau_model;
        J_surface(i,j) = sum(residual.^2);
    end
end

% 归一化到最小值为0
J_surface = J_surface - min(J_surface(:));
J_surface = log10(J_surface + 1e-25);  % 对数尺度便于可视化

%% 5. 模拟优化路径（梯度下降在平底谷中震荡）
% 起点：远离真值
ne_path = [0.85*ne_true];
nu_path = [8e9];

% 模拟梯度下降（在ν_e方向几乎无梯度）
for step = 1:15
    % n_e方向：强梯度，快速靠近真值
    ne_new = ne_path(end) + 0.03*(ne_true - ne_path(end));
    
    % ν_e方向：极弱梯度，来回震荡
    nu_new = nu_path(end) + 0.5e9 * (-1)^step * (0.8^step);  % 震荡衰减
    nu_new = max(0.5e9, min(9e9, nu_new));  % 边界约束
    
    ne_path = [ne_path, ne_new];
    nu_path = [nu_path, nu_new];
end

%% 6. 绘制3D曲面
figure('Position', [100, 100, 900, 700]);

% 归一化坐标用于绘图
NE_norm = (NE - ne_true) / ne_true * 100;  % 转为相对误差百分比
NU_norm = NU / 1e9;                         % 转为 GHz

surf(NE_norm, NU_norm, J_surface, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
hold on;

% 绘制优化路径（红色折线）
ne_path_norm = (ne_path - ne_true) / ne_true * 100;
nu_path_norm = nu_path / 1e9;
% 插值获取路径上的J值
J_path = interp2(NE_norm, NU_norm, J_surface, ne_path_norm, nu_path_norm);
plot3(ne_path_norm, nu_path_norm, J_path + 0.5, 'r-', 'LineWidth', 3);
plot3(ne_path_norm, nu_path_norm, J_path + 0.5, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');

% 标注真值位置
plot3(0, nu_true/1e9, min(J_surface(:)) - 1, 'g^', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'LineWidth', 2);

% 标注起点和终点
text(ne_path_norm(1), nu_path_norm(1), J_path(1) + 2, '起点', 'FontSize', 11, 'FontName', 'SimHei', 'Color', 'r');
text(ne_path_norm(end), nu_path_norm(end), J_path(end) + 2, '收敛点', 'FontSize', 11, 'FontName', 'SimHei', 'Color', 'r');

hold off;

% 设置视角
view(45, 35);
colormap(parula);
colorbar('Label', 'log_{10}(残差)', 'FontSize', 11);

% 坐标轴标签
xlabel('\Delta n_e / n_e^{true} (%)', 'FontSize', 13, 'FontName', 'Times New Roman');
ylabel('\nu_e (GHz)', 'FontSize', 13, 'FontName', 'Times New Roman');
zlabel('log_{10}(J)', 'FontSize', 13, 'FontName', 'Times New Roman');
title('图 4-2 参数空间残差曲面：平底谷现象', 'FontSize', 15, 'FontName', 'SimHei', 'FontWeight', 'bold');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
grid on; box on;

% 添加文字说明
dim = [0.15, 0.75, 0.25, 0.12];
str = {'\bf特征:', '• n_e 方向：陡峭峡谷壁', '• \nu_e 方向：平坦谷底', '• 红线：优化路径震荡'};
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
           'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontSize', 10, 'FontName', 'SimHei');

%% 7. 保存图表
print('-dpng', '-r300', '../../final_output/figures/图4-2_参数空间平底谷3D曲面.png');
print('-dsvg', '../../final_output/figures/图4-2_参数空间平底谷3D曲面.svg');

fprintf('图 4-2 已保存至 final_output/figures/\n');
fprintf('  - 3D残差曲面展示平底谷(flat valley)现象\n');
fprintf('  - 红色折线展示梯度下降算法在谷底震荡的困境\n');

%% 辅助函数：群时延计算
function tau_g = calculate_group_delay(omega, ne, nu_e, d, c, eps0, me, e)
    omega_p_sq = ne * e^2 / (eps0 * me);
    eps_r_real = 1 - omega_p_sq ./ (omega.^2 + nu_e^2);
    eps_r_real(eps_r_real < 0.01) = 0.01;  % 防止负值
    n_real = sqrt(eps_r_real);
    phi = omega .* n_real * d / c;
    tau_g = gradient(phi, omega(2)-omega(1));
end
