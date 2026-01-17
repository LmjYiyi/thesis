%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 参数敏感性分析：Lorentz模型与Butterworth滤波器模型
% 研究目标：分析各参数对群时延的影响，确定降维策略
% 创建日期：2026-01-18
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 0. 全局参数设置
c = 3e8;                     % 光速 (m/s)
d = 150e-3;                  % 介质厚度 (m)

% 输出路径
output_dir = fileparts(mfilename('fullpath'));

fprintf('===== 参数敏感性分析 =====\n');
fprintf('介质厚度 d = %.0f mm\n', d*1e3);

%% =========================================================================
%  第一部分：Lorentz模型敏感性分析
%  参数：f_res（谐振频率）、gamma（阻尼因子）
% =========================================================================

fprintf('\n===== Part 1: Lorentz模型敏感性分析 =====\n');

% 1.1 基准参数
f_res_base = 35.5e9;         % 谐振频率 (Hz)
gamma_base = 0.5e9;          % 阻尼因子 (Hz)
omega_p_meta = 2*pi*5e9;     % 等效等离子体角频率

% 探测频率范围
f_probe = linspace(34.2e9, 37.4e9, 500);

% 1.2 f_res 敏感性扫描（固定 gamma）
f_res_range = linspace(34.5e9, 36.5e9, 5);  % ±3%
gamma_fixed = gamma_base;

figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 400]);

subplot(1,2,1);
colors = lines(length(f_res_range));
hold on;
for i = 1:length(f_res_range)
    tau_g = calculate_lorentz_group_delay(f_probe, f_res_range(i), gamma_fixed, omega_p_meta, d, c);
    plot(f_probe/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('f_{res}=%.2f GHz', f_res_range(i)/1e9));
end
hold off;
xlabel('探测频率 (GHz)', 'FontName', 'SimHei');
ylabel('相对群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(a) f_{res} 变化 (γ=%.2f GHz固定)', gamma_fixed/1e9), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([34.2 37.4]);

% 1.3 gamma 敏感性扫描（固定 f_res）
gamma_range = linspace(0.1e9, 1.0e9, 5);  % ±100%
f_res_fixed = f_res_base;

subplot(1,2,2);
colors = lines(length(gamma_range));
hold on;
for i = 1:length(gamma_range)
    tau_g = calculate_lorentz_group_delay(f_probe, f_res_fixed, gamma_range(i), omega_p_meta, d, c);
    plot(f_probe/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('γ=%.2f GHz', gamma_range(i)/1e9));
end
hold off;
xlabel('探测频率 (GHz)', 'FontName', 'SimHei');
ylabel('相对群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(b) γ 变化 (f_{res}=%.2f GHz固定)', f_res_fixed/1e9), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([34.2 37.4]);

sgtitle('Lorentz模型参数敏感性分析', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');


% 1.4 敏感性量化：计算偏导数
fprintf('\n--- Lorentz模型敏感性量化 ---\n');

f_test = 35.5e9;  % 测试点：谐振频率处
delta_f_res = 0.01e9;  % 扰动量
delta_gamma = 0.01e9;

tau_base = calculate_lorentz_group_delay(f_test, f_res_base, gamma_base, omega_p_meta, d, c);
tau_df = calculate_lorentz_group_delay(f_test, f_res_base + delta_f_res, gamma_base, omega_p_meta, d, c);
tau_dg = calculate_lorentz_group_delay(f_test, f_res_base, gamma_base + delta_gamma, omega_p_meta, d, c);

sens_f_res = (tau_df - tau_base) / delta_f_res;  % ∂τ/∂f_res (s/Hz)
sens_gamma = (tau_dg - tau_base) / delta_gamma;  % ∂τ/∂γ (s/Hz)

% 归一化敏感性（相对变化）
rel_sens_f_res = sens_f_res * f_res_base / tau_base;  % (Δτ/τ) / (Δf_res/f_res)
rel_sens_gamma = sens_gamma * gamma_base / tau_base;  % (Δτ/τ) / (Δγ/γ)

fprintf('在 f=%.2f GHz 处:\n', f_test/1e9);
fprintf('  |∂τ/∂f_res| = %.3e s/Hz\n', abs(sens_f_res));
fprintf('  |∂τ/∂γ|     = %.3e s/Hz\n', abs(sens_gamma));
fprintf('  归一化敏感性比 |S_{f_res}| / |S_γ| = %.2f\n', abs(rel_sens_f_res/rel_sens_gamma));

%% =========================================================================
%  第二部分：Butterworth滤波器敏感性分析
%  参数：F0（中心频率）、BW（带宽）、N（阶数）
% =========================================================================

fprintf('\n===== Part 2: Butterworth滤波器敏感性分析 =====\n');

% 2.1 基准参数
F0_base = 14e9;              % 中心频率 (Hz)
BW_base = 8e9;               % 带宽 (Hz)
N_base = 5;                  % 阶数

% 探测频率范围
f_filter = linspace(10e9, 18e9, 500);

% 2.2 F0 敏感性扫描
F0_range = linspace(12e9, 16e9, 5);  % ±15%

figure(2); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1200 350]);

subplot(1,3,1);
colors = lines(length(F0_range));
hold on;
for i = 1:length(F0_range)
    tau_g = calculate_filter_group_delay(f_filter, F0_range(i), BW_base, N_base);
    plot(f_filter/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('F_0=%.0f GHz', F0_range(i)/1e9));
end
hold off;
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(a) F_0 变化 (BW=%d, N=%d)', BW_base/1e9, N_base), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([10 18]);

% 2.3 BW 敏感性扫描
BW_range = linspace(6e9, 10e9, 5);  % ±25%

subplot(1,3,2);
colors = lines(length(BW_range));
hold on;
for i = 1:length(BW_range)
    tau_g = calculate_filter_group_delay(f_filter, F0_base, BW_range(i), N_base);
    plot(f_filter/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('BW=%.0f GHz', BW_range(i)/1e9));
end
hold off;
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(b) BW 变化 (F_0=%d, N=%d)', F0_base/1e9, N_base), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([10 18]);

% 2.4 N 敏感性扫描
N_range = 2:2:8;  % 阶数 2,4,6,8

subplot(1,3,3);
colors = lines(length(N_range));
hold on;
for i = 1:length(N_range)
    tau_g = calculate_filter_group_delay(f_filter, F0_base, BW_base, N_range(i));
    plot(f_filter/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N=%d', N_range(i)));
end
hold off;
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(c) N 变化 (F_0=%d, BW=%d)', F0_base/1e9, BW_base/1e9), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([10 18]);

sgtitle('Butterworth滤波器参数敏感性分析', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');


% 2.5 敏感性量化
fprintf('\n--- Butterworth滤波器敏感性量化 ---\n');

f_test_f = 14e9;  % 测试点：中心频率处
delta_F0 = 0.01e9;
delta_BW = 0.01e9;
delta_N = 0.1;

tau_base_f = calculate_filter_group_delay(f_test_f, F0_base, BW_base, N_base);
tau_dF0 = calculate_filter_group_delay(f_test_f, F0_base + delta_F0, BW_base, N_base);
tau_dBW = calculate_filter_group_delay(f_test_f, F0_base, BW_base + delta_BW, N_base);
tau_dN = calculate_filter_group_delay(f_test_f, F0_base, BW_base, N_base + delta_N);

sens_F0 = (tau_dF0 - tau_base_f) / delta_F0;
sens_BW = (tau_dBW - tau_base_f) / delta_BW;
sens_N = (tau_dN - tau_base_f) / delta_N;

% 归一化敏感性
rel_sens_F0 = sens_F0 * F0_base / tau_base_f;
rel_sens_BW = sens_BW * BW_base / tau_base_f;
rel_sens_N = sens_N * N_base / tau_base_f;

fprintf('在 f=%.2f GHz 处:\n', f_test_f/1e9);
fprintf('  |∂τ/∂F0| = %.3e s/Hz, 归一化敏感性 S_{F0} = %.2f\n', abs(sens_F0), abs(rel_sens_F0));
fprintf('  |∂τ/∂BW| = %.3e s/Hz, 归一化敏感性 S_{BW} = %.2f\n', abs(sens_BW), abs(rel_sens_BW));
fprintf('  |∂τ/∂N|  = %.3e s,    归一化敏感性 S_N  = %.2f\n', abs(sens_N), abs(rel_sens_N));

%% =========================================================================
%  第三部分：Drude模型敏感性分析（等离子体）
%  参数：f_p（等离子体频率）、nue（碰撞频率）
% =========================================================================

fprintf('\n===== Part 3: Drude模型敏感性分析 =====\n');

% 3.1 基准参数
fp_base = 29e9;              % 等离子体频率 (Hz)
nue_base = 1.5e9;            % 碰撞频率 (Hz)

% 探测频率范围 (必须 > fp)
f_drude = linspace(30e9, 38e9, 500);

% 3.2 fp 敏感性扫描
fp_range = linspace(28e9, 30e9, 5);

figure(4); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 900 400]);

subplot(1,2,1);
colors = lines(length(fp_range));
hold on;
for i = 1:length(fp_range)
    tau_g = calculate_drude_group_delay(f_drude, fp_range(i), nue_base, d, c);
    plot(f_drude/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('f_p=%.1f GHz', fp_range(i)/1e9));
end
hold off;
xlabel('探测频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(a) f_p 变化 (ν_e=%.1f GHz)', nue_base/1e9), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([30 38]);

% 3.3 nue 敏感性扫描
nue_range = linspace(0.1e9, 5e9, 5);

subplot(1,2,2);
colors = lines(length(nue_range));
hold on;
for i = 1:length(nue_range)
    tau_g = calculate_drude_group_delay(f_drude, fp_base, nue_range(i), d, c);
    plot(f_drude/1e9, tau_g*1e9, 'Color', colors(i,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('ν_e=%.1f GHz', nue_range(i)/1e9));
end
hold off;
xlabel('探测频率 (GHz)', 'FontName', 'SimHei');
ylabel('群时延 (ns)', 'FontName', 'SimHei');
title(sprintf('(b) ν_e 变化 (f_p=%.1f GHz)', fp_base/1e9), 'FontName', 'SimHei');
legend('Location', 'northeast', 'FontSize', 8);
grid on; xlim([30 38]);

sgtitle('Drude模型参数敏感性分析', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

% 3.4 敏感性量化
fprintf('\n--- Drude模型敏感性量化 ---\n');
f_test_d = 34e9;
delta_fp = 0.01e9;
delta_nue = 0.01e9;

tau_base_d = calculate_drude_group_delay(f_test_d, fp_base, nue_base, d, c);
tau_dfp = calculate_drude_group_delay(f_test_d, fp_base + delta_fp, nue_base, d, c);
tau_dnue = calculate_drude_group_delay(f_test_d, fp_base, nue_base + delta_nue, d, c);

sens_fp = (tau_dfp - tau_base_d) / delta_fp;
sens_nue = (tau_dnue - tau_base_d) / delta_nue;

rel_sens_fp = sens_fp * fp_base / tau_base_d;
rel_sens_nue = sens_nue * nue_base / tau_base_d;

fprintf('在 f=%.2f GHz 处:\n', f_test_d/1e9);
fprintf('  |∂τ/∂fp|  = %.3e s/Hz\n', abs(sens_fp));
fprintf('  |∂τ/∂nue| = %.3e s/Hz\n', abs(sens_nue));
fprintf('  归一化敏感性比 |S_{fp}| / |S_{nue}| = %.2f\n', abs(rel_sens_fp/rel_sens_nue));

%% =========================================================================
%  第四部分：Jacobian条件数分析与降维策略
% =========================================================================

fprintf('\n===== Part 4: Jacobian条件数与降维策略 =====\n');

% 3.1 Lorentz模型：二参数Jacobian
f_sample = linspace(34.5e9, 37e9, 50);  % 采样点
J_lorentz = zeros(length(f_sample), 2);

for i = 1:length(f_sample)
    f_i = f_sample(i);
    tau0 = calculate_lorentz_group_delay(f_i, f_res_base, gamma_base, omega_p_meta, d, c);
    tau_df = calculate_lorentz_group_delay(f_i, f_res_base + delta_f_res, gamma_base, omega_p_meta, d, c);
    tau_dg = calculate_lorentz_group_delay(f_i, f_res_base, gamma_base + delta_gamma, omega_p_meta, d, c);
    J_lorentz(i, 1) = (tau_df - tau0) / delta_f_res * 1e10;  % 归一化
    J_lorentz(i, 2) = (tau_dg - tau0) / delta_gamma * 1e9;
end

cond_lorentz = cond(J_lorentz);
fprintf('Lorentz模型 Jacobian 条件数 = %.2f\n', cond_lorentz);

% 3.2 Butterworth模型：三参数Jacobian
J_filter = zeros(length(f_sample), 3);

for i = 1:length(f_sample)
    f_i = f_sample(i);
    tau0 = calculate_filter_group_delay(f_i, F0_base, BW_base, N_base);
    tau_dF = calculate_filter_group_delay(f_i, F0_base + delta_F0, BW_base, N_base);
    tau_dB = calculate_filter_group_delay(f_i, F0_base, BW_base + delta_BW, N_base);
    tau_dN = calculate_filter_group_delay(f_i, F0_base, BW_base, N_base + delta_N);
    J_filter(i, 1) = (tau_dF - tau0) / delta_F0 * 1e10;
    J_filter(i, 2) = (tau_dB - tau0) / delta_BW * 1e9;
    J_filter(i, 3) = (tau_dN - tau0) / delta_N;
end

cond_filter_3p = cond(J_filter);
cond_filter_2p = cond(J_filter(:, 1:2));  % 仅 F0, BW

fprintf('Butterworth 三参数 Jacobian 条件数 = %.2f\n', cond_filter_3p);
fprintf('Butterworth 二参数 (F0,BW) Jacobian 条件数 = %.2f\n', cond_filter_2p);

% 4.3 可视化：归一化敏感性对比
figure(5); clf;
set(gcf, 'Color', 'w', 'Position', [100 100 1200 350]);

subplot(1,3,1);
bar_data_lorentz = [abs(rel_sens_f_res), abs(rel_sens_gamma)];
bar(bar_data_lorentz, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', {'f_{res}', 'γ'}, 'FontSize', 12);
ylabel('归一化敏感性 |S|', 'FontName', 'SimHei');
title(sprintf('Lorentz (Cond=%.1f)', cond_lorentz), 'FontName', 'SimHei');
grid on;

subplot(1,3,2);
bar_data_drude = [abs(rel_sens_fp), abs(rel_sens_nue)];
bar(bar_data_drude, 'FaceColor', [0.4 0.8 0.4]);
set(gca, 'XTickLabel', {'f_p', 'ν_e'}, 'FontSize', 12);
ylabel('归一化敏感性 |S|', 'FontName', 'SimHei');
title(sprintf('Drude (|S_{fp}|/|S_{νe}| = %.0f)', abs(rel_sens_fp/rel_sens_nue)), 'FontName', 'SimHei');
grid on;

subplot(1,3,3);
bar_data_filter = [abs(rel_sens_F0), abs(rel_sens_BW), abs(rel_sens_N)];
bar(bar_data_filter, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTickLabel', {'F_0', 'BW', 'N'}, 'FontSize', 12);
ylabel('归一化敏感性 |S|', 'FontName', 'SimHei');
title(sprintf('Butterworth滤波器 (条件数=%.1f)', cond_filter_3p), 'FontName', 'SimHei');
grid on;

sgtitle('参数敏感性对比与降维策略依据', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');


%% =========================================================================
%  第五部分：降维策略结论
% =========================================================================

fprintf('\n===== 降维策略结论 =====\n');

fprintf('\n【Lorentz模型】\n');
if abs(rel_sens_f_res) / abs(rel_sens_gamma) > 5
    fprintf('  ✓ f_res 敏感性远大于 γ (比值=%.1f)\n', abs(rel_sens_f_res/rel_sens_gamma));
    fprintf('  ✓ 建议策略: 固定 γ 为先验值, 仅反演 f_res\n');
    fprintf('  ✓ 物理依据: γ 主要影响峰值高度, f_res 主导频率位置\n');
else
    fprintf('  △ 两参数敏感性相近, 建议同时反演\n');
end

fprintf('\n【Drude模型】\n');
if abs(rel_sens_fp) / abs(rel_sens_nue) > 10
    fprintf('  ✓ f_p 敏感性远大于 ν_e (比值=%.1f)\n', abs(rel_sens_fp/rel_sens_nue));
    fprintf('  ✓ 建议策略: 固定 ν_e, 仅反演 f_p (或 n_e)\n');
    fprintf('  ✓ 物理依据: ν_e 为二阶微扰 (ν_e/ω)^2\n');
else
    fprintf('  △ 敏感性比值较低, 需同时反演\n');
end

fprintf('\n【Butterworth滤波器模型】\n');
[~, min_idx] = min(bar_data_filter);
param_names = {'F_0', 'BW', 'N'};
fprintf('  敏感性排序: F_0=%.2f, BW=%.2f, N=%.2f\n', bar_data_filter(1), bar_data_filter(2), bar_data_filter(3));

if cond_filter_3p > 100
    fprintf('  ⚠ 三参数条件数过大 (%.1f), 存在病态问题\n', cond_filter_3p);
    fprintf('  ✓ 建议策略: 固定 %s, 仅反演其他两参数\n', param_names{min_idx});
else
    fprintf('  ✓ 三参数条件数可接受, 可同时反演\n');
end

fprintf('\n仿真完成！\n');

%% =========================================================================
%  局部函数
% =========================================================================

function tau_rel = calculate_lorentz_group_delay(f_vec, f_res, gamma, wp_meta, d, c)
    % Lorentz模型相对群时延计算
    % 输出: 相对群时延 = 超材料时延 - 真空时延
    
    % 处理标量输入（数值微分需要至少两个点）
    is_scalar = false;
    if isscalar(f_vec)
        is_scalar = true;
        f_vec = [f_vec, f_vec + 1e3]; % 增加 1kHz 偏移点
    end
    
    omega_vec = 2 * pi * f_vec;
    omega_res = 2 * pi * f_res;
    gamma_omega = 2 * pi * gamma;
    
    % Lorentz模型复介电常数
    eps_r = 1 + (wp_meta^2) ./ (omega_res^2 - omega_vec.^2 - 1i*gamma_omega*omega_vec);
    
    % 复波数
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 相位
    phi_meta = -real(k_vec) * d;
    
    % 数值微分求群时延
    d_phi = diff(phi_meta);
    d_omega = diff(omega_vec);
    tau_total = -d_phi ./ d_omega;
    
    if ~is_scalar
        tau_total = [tau_total, tau_total(end)];  % 维度补齐
    end
    
    % 减去真空时延
    tau_rel = tau_total - (d/c);
end

function tau_g = calculate_filter_group_delay(f_vec, F0, BW, N)
    % Butterworth滤波器群时延解析公式
    % 来源: LFMCW_filter_inversion_FINAL.m
    
    % 处理标量
    is_scalar = false;
    if isscalar(f_vec)
        is_scalar = true;
        f_vec = [f_vec, f_vec + 1e3];
    end
    
    x = (f_vec - F0) / (BW/2);
    tau_g = (2*N) / (pi*BW) .* (1 + x.^2).^(-(N+1)/2);
    
    if is_scalar
        tau_g = tau_g(1);
    end
end

function tau_rel = calculate_drude_group_delay(f_vec, fp, nue, d, c)
    % Drude模型相对群时延计算
    
    % 处理标量
    is_scalar = false;
    if isscalar(f_vec)
        is_scalar = true;
        f_vec = [f_vec, f_vec + 1e3];
    end
    
    omega = 2*pi*f_vec;
    wp = 2*pi*fp;
    gamma = 2*pi*nue;  % 碰撞频率 (rad/s)
    
    % 复介电常数 (Drude)
    % eps = 1 - wp^2 / (omega^2 + 1i*omega*gamma)
    % 注意：通常物理文献形式为 1 - wp^2/(omega*(omega+j*nue))
    % 这里复用 MATLAB 通用形式
    eps_r = 1 - (wp^2) ./ (omega.^2 + 1i.*omega.*gamma);
    
    % 复波数 k = w/c * sqrt(eps)
    k = (omega ./ c) .* sqrt(eps_r);
    
    % 相位 phi = -real(k)*d
    phi = -real(k) * d;
    
    % 群时延 tau = -dphi/dw
    dphi = diff(phi);
    dw = diff(omega);
    tau_total = -dphi ./ dw;
    
    if ~is_scalar
        tau_total = [tau_total, tau_total(end)]; 
    else
        tau_total = tau_total(1);
    end
    
    % 相对时延（减去真空）
    tau_rel = tau_total - (d/c);
end
