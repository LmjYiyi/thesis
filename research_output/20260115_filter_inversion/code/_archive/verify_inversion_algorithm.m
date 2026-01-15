%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 滤波器参数反演算法验证 - 简化版（绕过LFMCW传播建模）
% 目的：先验证反演算法本身是否正确，排除LFMCW信号处理的干扰
% 方法：直接生成理论群时延数据 + 噪声，然后反演
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% 滤波器真实参数
F0_true = 14e9;
BW_true = 8e9;
N_true = 5;

fprintf('===== 滤波器参数反演算法验证 =====\n');
fprintf('真实参数: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', F0_true/1e9, BW_true/1e9, N_true);

%% 1. 生成理论群时延数据（模拟"完美的ESPRIT测量"）

f_probe = linspace(10e9, 18e9, 100);  % 探测频率
tau_theory = calculate_filter_group_delay(f_probe, F0_true, BW_true, N_true);

% 添加噪声（模拟真实测量）
SNR_dB = 20;
noise_std = max(tau_theory) / (10^(SNR_dB/20));
tau_meas = tau_theory + noise_std * randn(size(tau_theory));

% 可视化测量数据
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 500]);
scatter(f_probe/1e9, tau_meas*1e9, 25, 'b', 'filled', 'DisplayName', '模拟测量数据');
hold on;
plot(f_probe/1e9, tau_theory*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '真实曲线');
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title('模拟群时延测量数据（理论+噪声）', 'FontName', 'SimHei');
legend('Location', 'northeast');
grid on;

%% 2. 三参数LM反演

fprintf('\n开始三参数LM反演...\n');

% 数据筛选（去掉边缘低信噪比点）
valid_mask = tau_meas > 0.1 * max(tau_meas);
X_fit = f_probe(valid_mask);
Y_fit = tau_meas(valid_mask);

% 权重（边缘增强）
tau_norm = Y_fit / max(Y_fit);
edge_enhance = 1 + 0.3 * (1 - tau_norm).^2;
Weights = (tau_norm .* edge_enhance).^2;

fprintf('有效数据点: %d\n', length(X_fit));

% 初始猜测
F0_guess = 13.5e9;
BW_guess = 9e9;
N_guess = 4;

fprintf('初始猜测: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', F0_guess/1e9, BW_guess/1e9, N_guess);

% 归一化
scale_F0 = 1e10;
scale_BW = 1e10;
scale_N = 1;
param_init = [F0_guess/scale_F0, BW_guess/scale_BW, N_guess];

% 残差函数
ResidualFunc = @(p) sqrt(Weights) .* ...
    (calculate_filter_group_delay(X_fit, p(1)*scale_F0, p(2)*scale_BW, p(3)) - Y_fit) * 1e9;

% 约束边界
lb = [10e9/scale_F0, 4e9/scale_BW, 2];
ub = [18e9/scale_F0, 12e9/scale_BW, 10];

% 优化
options = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', ...
    'Display', 'iter', 'MaxIterations', 100);

[param_opt, ~] = lsqnonlin(ResidualFunc, param_init, lb, ub, options);

F0_opt = param_opt(1) * scale_F0;
BW_opt = param_opt(2) * scale_BW;
N_opt = param_opt(3);

%% 3. 结果输出

fprintf('\n===== 反演结果 =====\n');
fprintf('真实值: F0=%.4fGHz, BW=%.4fGHz, N=%.1f\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('反演值: F0=%.4fGHz, BW=%.4fGHz, N=%.2f\n', F0_opt/1e9, BW_opt/1e9, N_opt);

err_F0 = (F0_opt - F0_true)/F0_true*100;
err_BW = (BW_opt - BW_true)/BW_true*100;
err_N = (N_opt - N_true)/N_true*100;

fprintf('误差: F0=%.2f%%, BW=%.2f%%, N=%.2f%%\n', err_F0, err_BW, err_N);
fprintf('整数阶数: N=%d\n', round(N_opt));
fprintf('====================\n');

%% 4. 可视化拟合结果

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 600]);

subplot(2,1,1);
scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
colorbar; ylabel(colorbar, '权重');
hold on;
plot(f_probe/1e9, tau_theory*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实曲线');

tau_fit = calculate_filter_group_delay(f_probe, F0_opt, BW_opt, N_opt);
plot(f_probe/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演曲线');

xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title_str = sprintf('反演结果 | F0误差:%.2f%%, BW误差:%.2f%%, N误差:%.2f%%', err_F0, err_BW, err_N);
title(title_str, 'FontName', 'SimHei');
legend('测量数据', '真实曲线', '反演曲线', 'Location', 'northeast');
grid on;

subplot(2,1,2);
residuals = Y_fit - calculate_filter_group_delay(X_fit, F0_opt, BW_opt, N_opt);
stem(X_fit/1e9, residuals*1e9, 'b', 'MarkerSize', 3);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontName', 'SimHei');
title('残差分布', 'FontName', 'SimHei');
grid on;

fprintf('\n仿真完成！\n');
fprintf('\n========================================\n');
fprintf('结论：\n');
if abs(err_F0) < 2 && abs(err_BW) < 5 && abs(err_N) < 10
    fprintf('✓ 反演算法本身是正确的！\n');
    fprintf('  问题出在LFMCW信号处理链的滤波器建模上。\n');
else
    fprintf('✗ 反演算法需要进一步调试。\n');
end
fprintf('========================================\n');

%% 局部函数

function tau_g = calculate_filter_group_delay(f_vec, F0, BW, N)
    x = (f_vec - F0) / (BW/2);
    tau_g = (2*N) / (pi*BW) .* (1 + x.^2).^(-(N+1)/2);
end
