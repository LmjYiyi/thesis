%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 带通滤波器三参数反演仿真 - (F0, B, N) 联合反演
% 研究课题：验证滤波器阶数N能否作为第三个反演参数
% 创建日期：2026-01-15
% 研究目标：实现中心频率F0、带宽B、阶数N的联合反演
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. 滤波器真实参数设置（来自datasheet）
F0_true = 14e9;           % 真实中心频率 (Hz)
B_true = 8e9;             % 真实带宽 (Hz)
N_true = 5;               % 真实滤波器阶数

fprintf('===== 滤波器三参数反演仿真 =====\n');
fprintf('真实参数：\n');
fprintf('  中心频率 F0 = %.2f GHz\n', F0_true/1e9);
fprintf('  带宽     B  = %.2f GHz\n', B_true/1e9);
fprintf('  阶数     N  = %d\n', N_true);
fprintf('================================\n\n');

%% 2. 生成理论群时延数据（模拟测量）
f_probe_start = 6e9;      % 探测起始频率（覆盖更宽范围以捕获边缘信息）
f_probe_end = 22e9;       % 探测终止频率
N_points = 300;           % 采样点数（增加以提高边缘分辨率）

f_probe = linspace(f_probe_start, f_probe_end, N_points);

% 计算真实群时延
tau_g_true = calculate_filter_delay(f_probe, F0_true, B_true, N_true);

% 添加测量噪声
SNR_dB = 25;              % 信噪比 (dB)
noise_power = max(tau_g_true)^2 / (10^(SNR_dB/10));
rng(42);                  % 固定随机种子以便复现
tau_g_meas = tau_g_true + sqrt(noise_power) * randn(size(tau_g_true));

fprintf('群时延数据生成完成\n');
fprintf('  峰值群时延: %.3f ns\n', max(tau_g_true)*1e9);
fprintf('  SNR: %d dB\n', SNR_dB);

%% 3. 可视化：测量数据

figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 500]);

scatter(f_probe/1e9, tau_g_meas*1e9, 15, 'b', 'filled', 'DisplayName', '模拟测量数据');
hold on;
plot(f_probe/1e9, tau_g_true*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '真实群时延');
xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('带通滤波器群时延测量数据（用于三参数反演）', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 11);
grid on; box on;
xlim([f_probe_start/1e9, f_probe_end/1e9]);

%% 4. 敏感性可视化：不同N对群时延的影响

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);

% N变化的影响（固定F0和B）
subplot(1,2,1);
N_variations = [2, 3, 5, 7, 10];
colors = lines(length(N_variations));
for i = 1:length(N_variations)
    tau_var = calculate_filter_delay(f_probe, F0_true, B_true, N_variations(i));
    plot(f_probe/1e9, tau_var*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N = %d', N_variations(i)));
    hold on;
end
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('阶数N变化的影响（固定F0=14GHz, B=8GHz）', 'FontSize', 13, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 9);
grid on;

% N和B联合影响（展示耦合效应）
subplot(1,2,2);
% 展示：N=3,B=6 和 N=6,B=12 可能产生相似峰值
param_sets = [
    3, 6e9;   % N=3, B=6GHz
    5, 10e9;  % N=5, B=10GHz（与真实值不同）
    5, 8e9;   % N=5, B=8GHz（真实值）
    7, 11.2e9 % N=7, B=11.2GHz（调整使峰值相近）
];
colors = lines(size(param_sets, 1));
for i = 1:size(param_sets, 1)
    N_i = param_sets(i, 1);
    B_i = param_sets(i, 2);
    tau_var = calculate_filter_delay(f_probe, F0_true, B_i, N_i);
    plot(f_probe/1e9, tau_var*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N=%d, B=%.1fGHz', N_i, B_i/1e9));
    hold on;
end
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('N和B的耦合效应演示', 'FontSize', 13, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 9);
grid on;

%% 5. 数据预处理与权重计算

fprintf('\n开始三参数反演 (Weighted LM Algorithm)...\n');

% 筛选有效数据（群时延大于峰值的5%，覆盖边缘区域）
tau_threshold = 0.05 * max(tau_g_meas);
valid_mask = tau_g_meas > tau_threshold;

X_fit = f_probe(valid_mask);
Y_fit = tau_g_meas(valid_mask);

% 权重策略：边缘区域增加权重以解耦N和B
% 使用V形权重：中心权重高，边缘权重也高
tau_norm = tau_g_meas(valid_mask) / max(tau_g_meas(valid_mask));
% 组合权重：基础权重 + 边缘增强
edge_enhance = 1 + 0.5 * (1 - tau_norm).^2;  % 边缘区域权重增益
W_raw = tau_norm .* edge_enhance;
Weights = (W_raw / max(W_raw)).^2;

fprintf('有效拟合数据点: %d / %d\n', sum(valid_mask), N_points);

%% 6. Levenberg-Marquardt 三参数反演

% 初始猜测值（故意偏离真实值）
F0_guess = 13e9;          % 偏低 (真实14GHz)
B_guess = 10e9;           % 偏宽 (真实8GHz)
N_guess = 3;              % 偏低 (真实5)

fprintf('初始猜测值：\n');
fprintf('  F0_guess = %.2f GHz (真实: %.2f GHz)\n', F0_guess/1e9, F0_true/1e9);
fprintf('  B_guess  = %.2f GHz (真实: %.2f GHz)\n', B_guess/1e9, B_true/1e9);
fprintf('  N_guess  = %d (真实: %d)\n', N_guess, N_true);

% 参数归一化（将不同量级参数映射到相近范围）
scale_F0 = 1e10;
scale_B = 1e10;
scale_N = 1;              % N本身就在1~10范围
param_init = [F0_guess/scale_F0, B_guess/scale_B, N_guess/scale_N];

% 构造残差函数
ResidualFunc = @(p) WeightedResiduals_3Param(p, scale_F0, scale_B, scale_N, X_fit, Y_fit, Weights);

% 设置优化选项
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'iter', ...
    'StepTolerance', 1e-10, ...
    'FunctionTolerance', 1e-10, ...
    'DiffMinChange', 0.001, ...
    'MaxIterations', 200);

% 执行优化
[param_opt, resnorm, ~, exitflag] = lsqnonlin(ResidualFunc, param_init, [], [], options);

% 还原物理参数
F0_opt = param_opt(1) * scale_F0;
B_opt = param_opt(2) * scale_B;
N_opt = param_opt(3) * scale_N;

%% 7. 结果输出与误差分析

fprintf('\n===== 三参数反演结果 =====\n');
fprintf('真实值    →  F0 = %.4f GHz,  B = %.4f GHz,  N = %.2f\n', F0_true/1e9, B_true/1e9, N_true);
fprintf('反演值    →  F0 = %.4f GHz,  B = %.4f GHz,  N = %.2f\n', F0_opt/1e9, B_opt/1e9, N_opt);

err_F0 = (F0_opt - F0_true) / F0_true * 100;
err_B = (B_opt - B_true) / B_true * 100;
err_N = (N_opt - N_true) / N_true * 100;

fprintf('相对误差  →  F0: %.2f%%,  B: %.2f%%,  N: %.2f%%\n', err_F0, err_B, err_N);
fprintf('===========================\n');

% 如果需要整数阶数，进行四舍五入
N_opt_int = round(N_opt);
fprintf('\n整数阶数（四舍五入）: N = %d\n', N_opt_int);

%% 8. 可视化拟合结果

figure(3); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 700]);

subplot(3,1,1);
% 测量数据（带颜色权重）
scatter(X_fit/1e9, Y_fit*1e9, 25, Weights, 'filled');
colorbar; ylabel(colorbar, '权重');
hold on;

% 真实曲线
plot(f_probe/1e9, tau_g_true*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实曲线');

% 拟合曲线（使用反演的N值，可能非整数）
tau_fit = calculate_filter_delay(f_probe, F0_opt, B_opt, N_opt);
plot(f_probe/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演曲线');

% 整数N的拟合曲线
tau_fit_int = calculate_filter_delay(f_probe, F0_opt, B_opt, N_opt_int);
plot(f_probe/1e9, tau_fit_int*1e9, 'm:', 'LineWidth', 2, 'DisplayName', sprintf('整数N=%d曲线', N_opt_int));

xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title_str = sprintf('三参数LM反演结果 | F0误差: %.2f%%, B误差: %.2f%%, N误差: %.2f%%', err_F0, err_B, err_N);
title(title_str, 'FontSize', 13, 'FontName', 'SimHei');
legend('测量数据', '真实曲线', '反演曲线', sprintf('整数N=%d曲线', N_opt_int), 'Location', 'northeast');
grid on; xlim([f_probe_start/1e9, f_probe_end/1e9]);

subplot(3,1,2);
% 残差分布
residuals = Y_fit - calculate_filter_delay(X_fit, F0_opt, B_opt, N_opt);
stem(X_fit/1e9, residuals*1e9, 'b', 'MarkerSize', 2);
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('残差分布', 'FontSize', 12, 'FontName', 'SimHei');
grid on; xlim([min(X_fit)/1e9, max(X_fit)/1e9]);

subplot(3,1,3);
% 权重分布
plot(X_fit/1e9, Weights, 'k', 'LineWidth', 1.5);
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('权重', 'FontSize', 11, 'FontName', 'SimHei');
title('拟合权重分布（边缘增强策略）', 'FontSize', 12, 'FontName', 'SimHei');
grid on; xlim([min(X_fit)/1e9, max(X_fit)/1e9]);

%% 9. 多初值测试（验证鲁棒性）

fprintf('\n===== 多初值测试 =====\n');

% 生成多组初值
initial_guesses = [
    12e9, 10e9, 3;   % 低估所有参数
    15e9, 6e9, 7;    % F0高估，B低估，N高估
    14e9, 8e9, 5;    % 接近真实值
    13e9, 12e9, 4;   % B严重高估
];

results = zeros(size(initial_guesses));
for i = 1:size(initial_guesses, 1)
    p0 = [initial_guesses(i,1)/scale_F0, initial_guesses(i,2)/scale_B, initial_guesses(i,3)/scale_N];
    [p_opt, ~] = lsqnonlin(ResidualFunc, p0, [], [], optimoptions('lsqnonlin', ...
        'Algorithm', 'levenberg-marquardt', 'Display', 'off', 'MaxIterations', 200));
    results(i,:) = [p_opt(1)*scale_F0/1e9, p_opt(2)*scale_B/1e9, p_opt(3)*scale_N];
end

fprintf('初值 (F0, B, N) → 反演结果 (F0, B, N)\n');
fprintf('真实值: (14.00, 8.00, 5)\n');
fprintf('--------------------------------------------\n');
for i = 1:size(initial_guesses, 1)
    fprintf('(%.1f, %.1f, %d) → (%.2f, %.2f, %.2f)\n', ...
        initial_guesses(i,1)/1e9, initial_guesses(i,2)/1e9, initial_guesses(i,3), ...
        results(i,1), results(i,2), results(i,3));
end
fprintf('--------------------------------------------\n');

fprintf('\n仿真完成！\n');

%% =========================================================================
%  局部函数定义
%% =========================================================================

function tau_g = calculate_filter_delay(f_vec, F0, B, N)
    % 计算Butterworth带通滤波器的群时延
    % 输入：
    %   f_vec - 频率向量 (Hz)
    %   F0    - 中心频率 (Hz)
    %   B     - 带宽 (Hz)
    %   N     - 滤波器阶数（可以是非整数）
    % 输出：
    %   tau_g - 群时延 (s)
    
    % 归一化频率偏移
    x = (f_vec - F0) / (B/2);
    
    % Butterworth带通群时延公式
    % tau_g(f) = (2N)/(pi*B) * [1 + x^2]^(-(N+1)/2)
    tau_g = (2*N) / (pi*B) .* (1 + x.^2).^(-(N+1)/2);
end

function F_vec = WeightedResiduals_3Param(p_scaled, scale_F0, scale_B, scale_N, f_data, tau_data, weights)
    % 三参数加权残差函数
    % 输入：
    %   p_scaled - 归一化参数 [F0/scale_F0, B/scale_B, N/scale_N]
    % 输出：
    %   F_vec - 加权残差向量
    
    % 还原物理参数
    F0_val = p_scaled(1) * scale_F0;
    B_val = p_scaled(2) * scale_B;
    N_val = p_scaled(3) * scale_N;
    
    % 物理约束检查
    if F0_val <= 0 || B_val <= 0 || N_val <= 0.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：F0应在探测范围内
    if F0_val < min(f_data)*0.3 || F0_val > max(f_data)*1.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：带宽应合理
    if B_val < 0.5e9 || B_val > 25e9
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：阶数应在合理范围
    if N_val < 1 || N_val > 15
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 计算理论群时延
    try
        tau_theory = calculate_filter_delay(f_data, F0_val, B_val, N_val);
        
        % 计算加权残差（乘以1e9转换为ns量级）
        F_vec = sqrt(weights) .* (tau_theory - tau_data) * 1e9;
        
        % NaN检查
        if any(isnan(F_vec)) || any(isinf(F_vec))
            F_vec = ones(size(f_data)) * 1e5;
        end
    catch
        F_vec = ones(size(f_data)) * 1e5;
    end
end
