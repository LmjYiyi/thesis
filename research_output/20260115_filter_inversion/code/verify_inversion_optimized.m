%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 滤波器参数反演算法验证 - 优化版（参考LM.m的技巧）
% 优化内容：
%   1. 更精细的加权策略（能量权重 + 边缘增强）
%   2. 参数归一化（防止梯度消失）
%   3. 更鲁棒的初值策略
%   4. 参考LM.m的残差函数设计
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% 滤波器真实参数
F0_true = 14e9;
BW_true = 8e9;
N_true = 5;

fprintf('===== 滤波器参数反演算法验证（优化版）=====\n');
fprintf('真实参数: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('优化策略: 能量加权 + 边缘增强 + 参数归一化\n');
fprintf('==========================================\n');

%% 1. 生成模拟测量数据（理论群时延 + 噪声 + 幅度信息）

% ===== 固定随机数种子，保证结果可重复 =====
rng(42);  % 使用固定种子42
fprintf('随机数种子已固定 (seed=42)，结果可重复\n\n');

f_probe = linspace(10e9, 18e9, 100);  % 探测频率
tau_theory = calculate_filter_group_delay(f_probe, F0_true, BW_true, N_true);

% 添加噪声（模拟真实测量）
SNR_dB = 20;
noise_std = max(tau_theory) / (10^(SNR_dB/20));
tau_meas = tau_theory + noise_std * randn(size(tau_theory));

% 模拟信号幅度（类似LM.m中ESPRIT提取的幅度）
% 假设信号幅度 ∝ 群时延（在通带内信号强，阻带弱）
amplitude_raw = tau_theory ./ max(tau_theory);
amplitude_meas = amplitude_raw + 0.05 * randn(size(amplitude_raw));  % 加幅度噪声
amplitude_meas(amplitude_meas < 0) = 0.01;  % 避免负值

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

%% 2. 三参数LM反演（参考LM.m的优化策略）

fprintf('\n开始三参数LM反演（优化版）...\n');

% -------------------------------------------------------------------------
% 2.1 数据筛选与准备（参考LM.m第423-435行）
% -------------------------------------------------------------------------
% 筛选数据：保留通带内且时延为正的点
fit_mask = (f_probe >= 10e9 + 0.05*(18e9-10e9)) & ...
           (f_probe <= 18e9 - 0.05*(18e9-10e9)) & ...
           (tau_meas > 1e-11);  % 过滤极小值

X_fit = f_probe(fit_mask);
Y_fit = tau_meas(fit_mask);
W_raw = amplitude_meas(fit_mask);  % 使用幅度作为权重

if isempty(X_fit)
    error('有效拟合数据点为空！');
end

% -------------------------------------------------------------------------
% 2.2 权重归一化（参考LM.m第442行，能量权重）
% -------------------------------------------------------------------------
% LM.m的策略：Weights = (W_raw / max(W_raw)).^2
% 这里在此基础上加入边缘增强策略
Weights_base = (W_raw / max(W_raw)).^2;  % 基础能量权重

% 边缘增强：增加边缘数据权重，帮助解耦B和N
tau_norm = Y_fit / max(Y_fit);
edge_factor = 1 + 0.5 * (1 - tau_norm).^2;  % 边缘处增益
Weights = Weights_base .* edge_factor;
Weights = Weights / max(Weights);  % 归一化到[0,1]

fprintf('有效数据点: %d\n', length(X_fit));
fprintf('权重策略: 能量权重 + 边缘增强\n');

% -------------------------------------------------------------------------
% 2.3 初始值策略（参考LM.m第447-450行）
% -------------------------------------------------------------------------
% LM.m策略：盲猜截止频率在探测起始频率的85%处
% 对滤波器：猜测中心频率略偏离真实值
F0_guess = 0.95 * min(X_fit) + 0.05 * max(X_fit);  % 靠近低频端
BW_guess = 1.1 * (max(X_fit) - min(X_fit));        % 略宽于探测范围
N_guess = 4;  % 略低于真实值

fprintf('初始猜测: F0=%.2fGHz, BW=%.2fGHz, N=%d\n', F0_guess/1e9, BW_guess/1e9, N_guess);

% -------------------------------------------------------------------------
% 2.4 参数归一化（参考LM.m第452-455行）
% -------------------------------------------------------------------------
% 将GHz量级参数映射到1左右，防止梯度消失
scale_F0 = 1e10;
scale_BW = 1e10;
scale_N = 1;
param_init = [F0_guess/scale_F0, BW_guess/scale_BW, N_guess/scale_N];

fprintf('参数归一化: scale_F0=1e10, scale_BW=1e10, scale_N=1\n');

% -------------------------------------------------------------------------
% 2.5 构造残差函数（参考LM.m第464行）
% -------------------------------------------------------------------------
% LM.m的残差函数：sqrt(weights) .* (tau_theory - tau_data) * 1e9
ResidualFunc = @(p) WeightedResiduals_Filter3P_Optimized(...
    p, scale_F0, scale_BW, scale_N, X_fit, Y_fit, Weights);

% -------------------------------------------------------------------------
% 2.6 设置优化选项（参考LM.m第469-474行）
% -------------------------------------------------------------------------
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
    'Display', 'iter', ...
    'StepTolerance', 1e-10, ...
    'FunctionTolerance', 1e-10, ...
    'DiffMinChange', 0.001, ...
    'MaxIterations', 200);

fprintf('\n开始Levenberg-Marquardt优化...\n');

% 执行优化
[param_opt, resnorm, ~, exitflag] = lsqnonlin(ResidualFunc, param_init, [], [], options);

% 还原物理参数
F0_opt = param_opt(1) * scale_F0;
BW_opt = param_opt(2) * scale_BW;
N_opt = param_opt(3) * scale_N;

%% 3. 结果输出

fprintf('\n===== 反演结果 =====\n');
fprintf('真实值: F0=%.4fGHz, BW=%.4fGHz, N=%.1f\n', F0_true/1e9, BW_true/1e9, N_true);
fprintf('反演值: F0=%.4fGHz, BW=%.4fGHz, N=%.2f\n', F0_opt/1e9, BW_opt/1e9, N_opt);

err_F0 = (F0_opt - F0_true)/F0_true*100;
err_BW = (BW_opt - BW_true)/BW_true*100;
err_N = (N_opt - N_true)/N_true*100;

fprintf('相对误差: F0=%.2f%%, BW=%.2f%%, N=%.2f%%\n', err_F0, err_BW, err_N);
fprintf('整数阶数: N=%d\n', round(N_opt));
fprintf('优化收敛标志: %d\n', exitflag);
fprintf('====================\n');

%% 4. 可视化拟合结果

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 700]);

% 主图：拟合对比
subplot(3,1,1);
scatter(X_fit/1e9, Y_fit*1e9, 30, Weights, 'filled');
colormap(jet);
c = colorbar; 
ylabel(c, '归一化权重', 'FontName', 'SimHei');
hold on;

plot(f_probe/1e9, tau_theory*1e9, 'r--', 'LineWidth', 1.5, 'DisplayName', '真实曲线');

tau_fit = calculate_filter_group_delay(f_probe, F0_opt, BW_opt, N_opt);
plot(f_probe/1e9, tau_fit*1e9, 'g-', 'LineWidth', 2.5, 'DisplayName', '反演曲线');

xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title_str = sprintf('优化反演结果 | F0误差:%.2f%%, BW误差:%.2f%%, N误差:%.2f%%', err_F0, err_BW, err_N);
title(title_str, 'FontName', 'SimHei');
legend('测量数据（颜色=权重）', '真实曲线', '反演曲线', 'Location', 'northeast');
grid on;

% 残差分布
subplot(3,1,2);
residuals = Y_fit - calculate_filter_group_delay(X_fit, F0_opt, BW_opt, N_opt);
stem(X_fit/1e9, residuals*1e9, 'b', 'MarkerSize', 3);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('残差 (ns)', 'FontName', 'SimHei');
title('残差分布', 'FontName', 'SimHei');
grid on;
yline(0, 'r--', 'LineWidth', 1.5);

% 权重分布
subplot(3,1,3);
plot(X_fit/1e9, Weights, 'k-', 'LineWidth', 1.5);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('归一化权重', 'FontName', 'SimHei');
title('权重分布（能量加权 + 边缘增强）', 'FontName', 'SimHei');
grid on;
ylim([0 1.1]);

fprintf('\n仿真完成！\n');

%% 5. 性能评估

fprintf('\n========================================\n');
fprintf('性能评估：\n');
fprintf('----------------------------------------\n');

% 评估各参数反演精度
if abs(err_F0) < 1
    fprintf('✓ F0反演精度: 优秀 (%.2f%%)\n', err_F0);
elseif abs(err_F0) < 3
    fprintf('⚠ F0反演精度: 良好 (%.2f%%)\n', err_F0);
else
    fprintf('✗ F0反演精度: 需改进 (%.2f%%)\n', err_F0);
end

if abs(err_BW) < 5
    fprintf('✓ BW反演精度: 优秀 (%.2f%%)\n', err_BW);
elseif abs(err_BW) < 15
    fprintf('⚠ BW反演精度: 良好 (%.2f%%)\n', err_BW);
else
    fprintf('✗ BW反演精度: 需改进 (%.2f%%)\n', err_BW);
end

if abs(err_N) < 10
    fprintf('✓ N反演精度: 优秀 (%.2f%%)\n', err_N);
elseif abs(err_N) < 20
    fprintf('⚠ N反演精度: 良好 (%.2f%%)\n', err_N);
else
    fprintf('✗ N反演精度: 需改进 (%.2f%%)\n', err_N);
end

fprintf('----------------------------------------\n');
fprintf('总体结论：\n');
if abs(err_F0) < 1 && abs(err_BW) < 15 && abs(err_N) < 20
    fprintf('✓ 优化后的反演算法性能优秀！\n');
    fprintf('  可用于滤波器参数诊断。\n');
else
    fprintf('⚠ 反演算法需要进一步优化。\n');
end
fprintf('========================================\n');

%% =========================================================================
%  局部函数定义
%% =========================================================================

function tau_g = calculate_filter_group_delay(f_vec, F0, BW, N)
    % 计算Butterworth带通滤波器的群时延
    x = (f_vec - F0) / (BW/2);
    tau_g = (2*N) / (pi*BW) .* (1 + x.^2).^(-(N+1)/2);
end

function F_vec = WeightedResiduals_Filter3P_Optimized(p_scaled, scale_F0, scale_BW, scale_N, f_data, tau_data, weights)
    % 优化版加权残差函数（参考LM.m的实现）
    % 输入：
    %   p_scaled - 归一化参数 [F0/scale_F0, BW/scale_BW, N/scale_N]
    %   weights  - 归一化权重（能量 + 边缘增强）
    % 输出：
    %   F_vec - 加权残差向量
    
    % 还原物理参数
    F0_val = p_scaled(1) * scale_F0;
    BW_val = p_scaled(2) * scale_BW;
    N_val = p_scaled(3) * scale_N;
    
    % 物理约束检查
    if F0_val <= 0 || BW_val <= 0 || N_val <= 0.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：F0应在探测范围内
    if F0_val < min(f_data)*0.5 || F0_val > max(f_data)*1.5
        F_vec = ones(size(f_data)) * 1e5;
        return;
    end
    
    % 约束：带宽应合理
    if BW_val < 1e9 || BW_val > 20e9
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
        tau_theory = calculate_filter_group_delay(f_data, F0_val, BW_val, N_val);
        
        % 计算加权残差（参考LM.m第553行）
        % 乘以1e9转换为ns量级，防止数值过小
        F_vec = sqrt(weights) .* (tau_theory - tau_data) * 1e9;
        
        % NaN检查
        if any(isnan(F_vec)) || any(isinf(F_vec))
            F_vec = ones(size(f_data)) * 1e5;
        end
    catch
        F_vec = ones(size(f_data)) * 1e5;
    end
end
