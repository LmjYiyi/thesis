%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 带通滤波器参数反演验证性仿真
% 研究课题：利用群时延特征反演滤波器中心频率和带宽
% 创建日期：2026-01-15
% 参考代码：LM.m（等离子体诊断算法）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. 滤波器参数设置（真实值，待反演）
fprintf('========================================\n');
fprintf('带通滤波器参数反演验证性仿真\n');
fprintf('========================================\n\n');

% 目标滤波器参数（真实值）
F0_true = 14e9;          % 中心频率 14 GHz
B_true = 8e9;            % 带宽 8 GHz (10-18 GHz)
f_low_true = F0_true - B_true/2;   % 下边频 10 GHz
f_high_true = F0_true + B_true/2;  % 上边频 18 GHz
N_order = 5;             % 滤波器阶数（假设5阶Butterworth）

% 探测频率范围
f_probe = linspace(8e9, 20e9, 500);  % 8-20 GHz
omega_probe = 2*pi*f_probe;

fprintf('目标滤波器参数：\n');
fprintf('  中心频率 F0 = %.2f GHz\n', F0_true/1e9);
fprintf('  带宽 B = %.2f GHz\n', B_true/1e9);
fprintf('  通带范围：%.2f - %.2f GHz\n', f_low_true/1e9, f_high_true/1e9);

%% 2. 带通滤波器群时延理论模型

% =========================================================================
% 理论推导：N阶Butterworth带通滤波器群时延
% =========================================================================
% 对于归一化低通原型 H_LP(s)，其群时延为：
%   τ_g(ω) = -d(arg H)/dω
%
% 对于带通变换 s → (s² + ω₀²)/(B·s)，其中：
%   ω₀ = 2π·F₀ (中心角频率)
%   B = 2π·(带宽) (带宽角频率)
%
% N阶Butterworth带通滤波器的群时延近似公式：
%   τ_g(f) ≈ N/(π·B) · 1/[1 + ((f-F0)/(B/2))²]^((N+1)/2)
%         + N/(π·B) · 1/[1 + ((f+F0)/(B/2))²]^((N+1)/2)
%
% 简化后（忽略负频率项，仅考虑正频率通带）：
%   τ_g(f) ≈ τ_peak · [1 + ((f-F0)/(B/2))²]^(-(N+1)/2)
%
% 其中峰值时延 τ_peak ≈ 2N/(π·B)
% =========================================================================

% 定义群时延计算函数
calculate_filter_delay = @(f, F0, B, N) ...
    (2*N./(pi*B)) .* (1 + ((f - F0)./(B/2)).^2).^(-(N+1)/2);

% 计算理论群时延（使用真实参数）
tau_theory_true = calculate_filter_delay(f_probe, F0_true, B_true, N_order);

% 峰值时延
tau_peak_theory = 2*N_order / (pi*B_true);
fprintf('\n理论峰值群时延 τ_peak = %.3f ns\n', tau_peak_theory*1e9);

%% 3. 生成"测量"数据（添加噪声模拟真实测量）

% 添加高斯噪声
SNR_dB = 30;  % 信噪比 30 dB
noise_power = max(tau_theory_true)^2 / (10^(SNR_dB/10));
noise = sqrt(noise_power) * randn(size(tau_theory_true));
tau_meas = tau_theory_true + noise;

% 模拟测量点（稀疏采样）
N_samples = 50;  % 测量点数
sample_idx = round(linspace(1, length(f_probe), N_samples));
f_samples = f_probe(sample_idx);
tau_samples = tau_meas(sample_idx);

fprintf('测量数据：%d 个采样点，SNR = %d dB\n', N_samples, SNR_dB);

%% 4. 参数反演：借鉴LM.m的加权最小二乘算法

fprintf('\n========================================\n');
fprintf('开始参数反演 (Levenberg-Marquardt)...\n');
fprintf('========================================\n');

% 初始猜测（偏离真实值）
F0_guess = 13e9;   % 猜测 13 GHz (真实 14 GHz)
B_guess = 6e9;     % 猜测 6 GHz (真实 8 GHz)

fprintf('初始猜测: F0 = %.2f GHz, B = %.2f GHz\n', F0_guess/1e9, B_guess/1e9);

% 参数归一化
scale_F0 = 1e10;
scale_B = 1e9;
param_init = [F0_guess/scale_F0, B_guess/scale_B];

% 权重设计：峰值附近权重更高
weights = tau_samples / max(tau_samples);
weights = weights.^2;

% 定义残差函数
residual_func = @(p) weighted_residual_filter(p, [scale_F0, scale_B], ...
    N_order, f_samples, tau_samples, weights);

% LM优化设置
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'iter', 'StepTolerance', 1e-8, 'FunctionTolerance', 1e-10, ...
    'MaxIterations', 100);

% 边界约束（归一化后）
lb = [0.5*min(f_samples)/scale_F0, 1e9/scale_B];
ub = [2*max(f_samples)/scale_F0, 15e9/scale_B];

% 执行优化
[param_opt, resnorm, ~, exitflag] = lsqnonlin(residual_func, param_init, lb, ub, options);

% 还原物理参数
F0_opt = param_opt(1) * scale_F0;
B_opt = param_opt(2) * scale_B;

%% 5. 结果输出与验证

fprintf('\n========================================\n');
fprintf('反演结果\n');
fprintf('========================================\n');

fprintf('\n中心频率 F0:\n');
fprintf('  真实值: %.4f GHz\n', F0_true/1e9);
fprintf('  反演值: %.4f GHz\n', F0_opt/1e9);
fprintf('  相对误差: %.2f%%\n', abs(F0_opt - F0_true)/F0_true * 100);

fprintf('\n带宽 B:\n');
fprintf('  真实值: %.4f GHz\n', B_true/1e9);
fprintf('  反演值: %.4f GHz\n', B_opt/1e9);
fprintf('  相对误差: %.2f%%\n', abs(B_opt - B_true)/B_true * 100);

%% 6. 可视化

figure(1); clf;
set(gcf, 'Position', [100, 100, 1000, 800], 'Color', 'w');

% 子图1：群时延曲线对比
subplot(2,2,[1,2]);
hold on;
plot(f_probe/1e9, tau_theory_true*1e9, 'b-', 'LineWidth', 2, ...
    'DisplayName', '理论真实值');
scatter(f_samples/1e9, tau_samples*1e9, 40, 'ro', 'filled', ...
    'DisplayName', '模拟测量点');

% 计算反演后的拟合曲线
tau_fit = calculate_filter_delay(f_probe, F0_opt, B_opt, N_order);
plot(f_probe/1e9, tau_fit*1e9, 'g--', 'LineWidth', 2.5, ...
    'DisplayName', '反演拟合曲线');

xline(F0_true/1e9, 'b:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xline(F0_opt/1e9, 'g:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

grid on; box on;
xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title(sprintf('带通滤波器群时延反演结果 (F_0误差: %.2f%%, B误差: %.2f%%)', ...
    abs(F0_opt-F0_true)/F0_true*100, abs(B_opt-B_true)/B_true*100), ...
    'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'best', 'FontSize', 11);
xlim([8, 20]);

% 子图2：残差分析
subplot(2,2,3);
tau_samples_fit = calculate_filter_delay(f_samples, F0_opt, B_opt, N_order);
residuals = (tau_samples - tau_samples_fit) * 1e9;
bar(f_samples/1e9, residuals, 'FaceColor', [0.5 0.5 0.5]);
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontSize', 11, 'FontName', 'SimHei');
title('拟合残差分布', 'FontSize', 12, 'FontName', 'SimHei');
grid on;

% 子图3：权重分布
subplot(2,2,4);
bar(f_samples/1e9, weights, 'FaceColor', [0.2 0.6 0.8]);
xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', 'SimHei');
ylabel('权重', 'FontSize', 11, 'FontName', 'SimHei');
title('各频率点的拟合权重', 'FontSize', 12, 'FontName', 'SimHei');
grid on;

%% 7. 敏感性分析：参数对群时延的影响

fprintf('\n========================================\n');
fprintf('参数敏感性分析\n');
fprintf('========================================\n');

% F0 敏感性
F0_test = [13e9, 14e9, 15e9];
figure(2); clf;
set(gcf, 'Position', [200, 200, 900, 400], 'Color', 'w');

subplot(1,2,1);
colors = lines(3);
hold on;
for i = 1:3
    tau_test = calculate_filter_delay(f_probe, F0_test(i), B_true, N_order);
    plot(f_probe/1e9, tau_test*1e9, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', sprintf('F_0 = %.0f GHz', F0_test(i)/1e9));
end
grid on; box on;
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title('中心频率 F_0 对群时延的影响 (固定B=8GHz)', 'FontName', 'SimHei');
legend('Location', 'best');

% B 敏感性
B_test = [6e9, 8e9, 10e9];
subplot(1,2,2);
hold on;
for i = 1:3
    tau_test = calculate_filter_delay(f_probe, F0_true, B_test(i), N_order);
    plot(f_probe/1e9, tau_test*1e9, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', sprintf('B = %.0f GHz', B_test(i)/1e9));
end
grid on; box on;
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title('带宽 B 对群时延的影响 (固定F_0=14GHz)', 'FontName', 'SimHei');
legend('Location', 'best');

% 计算敏感度指标
delta_F0 = 1e9;  % F0 变化 1 GHz
delta_B = 2e9;   % B 变化 2 GHz
idx_center = find(abs(f_probe - F0_true) < 0.5e9, 1);

tau_F0_plus = calculate_filter_delay(f_probe, F0_true+delta_F0, B_true, N_order);
tau_F0_minus = calculate_filter_delay(f_probe, F0_true-delta_F0, B_true, N_order);
S_F0 = abs(tau_F0_plus(idx_center) - tau_F0_minus(idx_center)) / (2*delta_F0/F0_true) / tau_theory_true(idx_center);

tau_B_plus = calculate_filter_delay(f_probe, F0_true, B_true+delta_B, N_order);
tau_B_minus = calculate_filter_delay(f_probe, F0_true, B_true-delta_B, N_order);
S_B = abs(tau_B_plus(idx_center) - tau_B_minus(idx_center)) / (2*delta_B/B_true) / tau_theory_true(idx_center);

fprintf('敏感度指标（在中心频率处）：\n');
fprintf('  S_τ^F0 = %.2f\n', S_F0);
fprintf('  S_τ^B = %.2f\n', S_B);
fprintf('  敏感度比值: %.2f\n', S_F0/S_B);

%% 8. 保存结果

fprintf('\n========================================\n');
fprintf('仿真完成！\n');
fprintf('========================================\n');

% 结论输出
if abs(F0_opt - F0_true)/F0_true < 0.01 && abs(B_opt - B_true)/B_true < 0.05
    fprintf('\n【结论】✅ 反演成功！算法能够从群时延数据中准确提取滤波器参数。\n');
    fprintf('  - 中心频率 F0 反演精度 < 1%%\n');
    fprintf('  - 带宽 B 反演精度 < 5%%\n');
    fprintf('  - LM.m 算法框架可成功迁移到滤波器参数反演应用！\n');
else
    fprintf('\n【结论】⚠️ 反演精度有待提高，需进一步优化算法或模型。\n');
end

% =========================================================================
% 局部函数
% =========================================================================

function F_vec = weighted_residual_filter(params, scales, N, f_data, tau_data, weights)
    % 还原物理参数
    F0 = params(1) * scales(1);
    B = params(2) * scales(2);
    
    % 物理约束
    if F0 <= 0 || B <= 0 || B > 2*F0
        F_vec = ones(size(f_data)) * 1e10;
        return;
    end
    
    % 计算理论群时延
    tau_theory = (2*N./(pi*B)) .* (1 + ((f_data - F0)./(B/2)).^2).^(-(N+1)/2);
    
    % 加权残差
    F_vec = sqrt(weights) .* (tau_theory - tau_data) * 1e9;
    
    if any(isnan(F_vec))
        F_vec = ones(size(f_data)) * 1e10;
    end
end
