%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 带通滤波器参数反演仿真 - 基于LM.m算法迁移
% 研究课题：使用Levenberg-Marquardt算法反演滤波器中心频率和带宽
% 创建日期：2026-01-15
% 研究目标：验证LM.m反演算法能否成功提取带通滤波器的F0和B参数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. 滤波器真实参数设置（来自datasheet）
% 滤波器规格：
% - 中心频率 F0 = 14 GHz
% - 通带范围 10-18 GHz → 带宽 B = 8 GHz
% - 通带插入损耗 ≤ 2.0 dB
% - 通带纹波 ≤ 1.2 dB

F0_true = 14e9;           % 真实中心频率 (Hz)
B_true = 8e9;             % 真实带宽 (Hz)
N_order = 5;              % 滤波器阶数（假设值，基于通带特性估计）

fprintf('===== 滤波器参数反演仿真 =====\n');
fprintf('真实参数：\n');
fprintf('  中心频率 F0 = %.2f GHz\n', F0_true/1e9);
fprintf('  带宽     B  = %.2f GHz\n', B_true/1e9);
fprintf('  阶数     N  = %d\n', N_order);
fprintf('============================\n\n');

%% 2. 生成理论群时延曲线（作为"测量数据"）
% 探测频率范围：覆盖通带及部分阻带

f_probe_start = 8e9;      % 探测起始频率
f_probe_end = 20e9;       % 探测终止频率
N_points = 200;           % 采样点数

f_probe = linspace(f_probe_start, f_probe_end, N_points);

% 计算理论群时延（基于Butterworth带通滤波器模型）
% tau_g(f) = (2N)/(pi*B) * [1 + ((f-F0)/(B/2))^2]^(-(N+1)/2)
tau_g_true = calculate_filter_delay(f_probe, F0_true, B_true, N_order);

% 添加测量噪声（模拟真实测量情况）
SNR_dB = 30;              % 信噪比 (dB)
noise_power = max(tau_g_true)^2 / (10^(SNR_dB/10));
tau_g_meas = tau_g_true + sqrt(noise_power) * randn(size(tau_g_true));

fprintf('群时延数据生成完成\n');
fprintf('  峰值群时延: %.3f ns (在 f = F0 处)\n', max(tau_g_true)*1e9);
fprintf('  添加噪声SNR: %d dB\n', SNR_dB);

%% 3. 可视化：测量数据与真实曲线

figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 500]);

scatter(f_probe/1e9, tau_g_meas*1e9, 20, 'b', 'filled', 'DisplayName', '模拟测量数据');
hold on;
plot(f_probe/1e9, tau_g_true*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '真实群时延');
xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('带通滤波器群时延测量数据', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 11);
grid on; box on;
xlim([f_probe_start/1e9, f_probe_end/1e9]);

%% 4. 数据预处理与权重计算
fprintf('\n开始参数反演 (Weighted LM Algorithm)...\n');

% 筛选通带内数据（群时延较大的区域）
tau_threshold = 0.1 * max(tau_g_meas);  % 取峰值的10%作为阈值
valid_mask = tau_g_meas > tau_threshold;

X_fit = f_probe(valid_mask);
Y_fit = tau_g_meas(valid_mask);

% 权重设计：群时延越大，权重越高（类似LM.m中的能量加权）
W_raw = tau_g_meas(valid_mask);
Weights = (W_raw / max(W_raw)).^2;

fprintf('有效拟合数据点: %d / %d\n', sum(valid_mask), N_points);

%% 5. Levenberg-Marquardt 反演
% 初始猜测值（偏离真实值测试收敛性）
F0_guess = 12e9;          % 初始猜测中心频率 (偏低)
B_guess = 10e9;           % 初始猜测带宽 (偏宽)

fprintf('初始猜测值：\n');
fprintf('  F0_guess = %.2f GHz (真实: %.2f GHz)\n', F0_guess/1e9, F0_true/1e9);
fprintf('  B_guess  = %.2f GHz (真实: %.2f GHz)\n', B_guess/1e9, B_true/1e9);

% 参数归一化（将GHz量级参数映射到1左右）
scale_F0 = 1e10;
scale_B = 1e10;
param_init = [F0_guess/scale_F0, B_guess/scale_B];

% 构造残差函数
ResidualFunc = @(p) WeightedResiduals_Filter(p, scale_F0, scale_B, N_order, X_fit, Y_fit, Weights);

% 设置优化选项（参考LM.m）
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'iter', ...
    'StepTolerance', 1e-8, ...
    'FunctionTolerance', 1e-8, ...
    'DiffMinChange', 0.001, ...
    'MaxIterations', 100);

% 执行优化
[param_opt, resnorm, ~, exitflag] = lsqnonlin(ResidualFunc, param_init, [], [], options);

% 还原物理参数
F0_opt = param_opt(1) * scale_F0;
B_opt = param_opt(2) * scale_B;

%% 6. 结果输出与误差分析

fprintf('\n===== 反演结果 =====\n');
fprintf('真实值    →  F0 = %.4f GHz,  B = %.4f GHz\n', F0_true/1e9, B_true/1e9);
fprintf('反演值    →  F0 = %.4f GHz,  B = %.4f GHz\n', F0_opt/1e9, B_opt/1e9);

err_F0 = (F0_opt - F0_true) / F0_true * 100;
err_B = (B_opt - B_true) / B_true * 100;

fprintf('相对误差  →  F0: %.2f%%,  B: %.2f%%\n', err_F0, err_B);
fprintf('=========================\n');

%% 7. 可视化拟合结果

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 600]);

subplot(2,1,1);
% 测量数据（带颜色权重）
scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
colorbar; ylabel(colorbar, '权重');
hold on;

% 真实曲线
plot(f_probe/1e9, tau_g_true*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实曲线');

% 拟合曲线
tau_fit = calculate_filter_delay(f_probe, F0_opt, B_opt, N_order);
plot(f_probe/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演曲线');

xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title_str = sprintf('加权LM反演结果 | F0误差: %.2f%%, B误差: %.2f%%', err_F0, err_B);
title(title_str, 'FontSize', 14, 'FontName', 'SimHei');
legend('测量数据', '真实曲线', '反演曲线', 'Location', 'northeast');
grid on; xlim([f_probe_start/1e9, f_probe_end/1e9]);

subplot(2,1,2);
% 残差分布
residuals = Y_fit - calculate_filter_delay(X_fit, F0_opt, B_opt, N_order);
stem(X_fit/1e9, residuals*1e9, 'b', 'MarkerSize', 3);
xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('残差分布', 'FontSize', 12, 'FontName', 'SimHei');
grid on; xlim([min(X_fit)/1e9, max(X_fit)/1e9]);

%% 8. 敏感性分析（参数变化对群时延的影响）

figure(3); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 400]);

% F0变化的影响
subplot(1,2,1);
F0_variations = F0_true * [0.9, 0.95, 1.0, 1.05, 1.1];
colors = lines(length(F0_variations));
for i = 1:length(F0_variations)
    tau_var = calculate_filter_delay(f_probe, F0_variations(i), B_true, N_order);
    plot(f_probe/1e9, tau_var*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('F0 = %.1f GHz', F0_variations(i)/1e9));
    hold on;
end
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('中心频率F0变化的影响', 'FontSize', 13, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 9);
grid on;

% B变化的影响
subplot(1,2,2);
B_variations = B_true * [0.6, 0.8, 1.0, 1.2, 1.4];
colors = lines(length(B_variations));
for i = 1:length(B_variations)
    tau_var = calculate_filter_delay(f_probe, F0_true, B_variations(i), N_order);
    plot(f_probe/1e9, tau_var*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('B = %.1f GHz', B_variations(i)/1e9));
    hold on;
end
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('带宽B变化的影响', 'FontSize', 13, 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 9);
grid on;

fprintf('\n仿真完成！图表已生成。\n');

%% =========================================================================
%  局部函数定义
%% =========================================================================

function tau_g = calculate_filter_delay(f_vec, F0, B, N)
    % 计算Butterworth带通滤波器的群时延
    % 输入：
    %   f_vec - 频率向量 (Hz)
    %   F0    - 中心频率 (Hz)
    %   B     - 带宽 (Hz)
    %   N     - 滤波器阶数
    % 输出：
    %   tau_g - 群时延 (s)
    
    % 归一化频率偏移
    x = (f_vec - F0) / (B/2);
    
    % Butterworth带通群时延公式
    % tau_g(f) = (2N)/(pi*B) * [1 + x^2]^(-(N+1)/2)
    tau_g = (2*N) / (pi*B) .* (1 + x.^2).^(-(N+1)/2);
end

function F_vec = WeightedResiduals_Filter(p_scaled, scale_F0, scale_B, N, f_data, tau_data, weights)
    % 计算加权残差向量
    % 输入：
    %   p_scaled - 归一化参数 [F0/scale_F0, B/scale_B]
    %   其他参数同上
    % 输出：
    %   F_vec - 加权残差向量
    
    % 还原物理参数
    F0_val = p_scaled(1) * scale_F0;
    B_val = p_scaled(2) * scale_B;
    
    % 物理约束检查
    if F0_val <= 0 || B_val <= 0
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：F0应在探测范围内
    if F0_val < min(f_data)*0.5 || F0_val > max(f_data)*1.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：带宽应合理（不能太窄或太宽）
    if B_val < 0.5e9 || B_val > 20e9
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 计算理论群时延
    try
        tau_theory = calculate_filter_delay(f_data, F0_val, B_val, N);
        
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
