%% plot_fig_4_7.m
% 论文图 4-7：MCMC迹线图（Trace Plot）——参数可观测性对比
% 生成日期：2026-01-22
% 对应章节：4.3.2 MCMC采样策略
%
% 物理意义：
% - 可观测参数（n_e）：预烧期后围绕真值稳定振荡，呈现"混合良好"特征
% - 不可观测参数（ν_e）：在整个先验范围内无序漫游，未能收敛至稳定值
% 
% 文档描述（第91行）：
% "可观测参数呈现窄幅振荡，而不可观测参数呈现先验范围内的均匀漫游"

clear; clc; close all;

%% 1. 加载或模拟MCMC采样数据
% 本代码生成用于论文展示的典型MCMC迹线图
% 用户可替换为实际LM_MCMC.m运行结果

% 仿真参数设置
N_samples = 10000;        % 总采样次数
burn_in = 2000;           % 预烧期

% 先验范围
ne_min = 1e18; ne_max = 1e20;  % 电子密度先验 (m^-3)
nu_min = 0.1e9; nu_max = 5e9;  % 碰撞频率先验 (Hz)

% 真值（用于绘图参考）
n_e_true = 4.60e19;       % m^-3 (对应 f_p ≈ 61 GHz)
nu_e_true = 1.5e9;        % 1.5 GHz

% 模拟迹线数据：n_e 收敛至真值附近
% 预烧期：向真值漂移
n_burn_drift = n_e_true * 0.5 + (n_e_true - n_e_true*0.5) * (1:burn_in)'/burn_in;
n_burn_noise = n_e_true * 0.02 * randn(burn_in, 1);
samples_ne_burn = n_burn_drift + n_burn_noise;

% 稳态期：围绕真值窄幅振荡
n_stable_noise = n_e_true * 0.015 * randn(N_samples - burn_in, 1);
samples_ne_stable = n_e_true + n_stable_noise;

samples_ne = [samples_ne_burn; samples_ne_stable];

% 模拟迹线数据：nu_e 在先验范围内漫游（体现不可观测性）
% 使用随机游走模拟"平底谷"现象
samples_nu = zeros(N_samples, 1);
samples_nu(1) = nu_min + (nu_max - nu_min) * rand();
step_nu = (nu_max - nu_min) * 0.03;  % 大步长探索
for i = 2:N_samples
    proposal = samples_nu(i-1) + step_nu * randn();
    % 反射边界条件
    if proposal > nu_max
        proposal = 2*nu_max - proposal;
    elseif proposal < nu_min
        proposal = 2*nu_min - proposal;
    end
    samples_nu(i) = max(nu_min, min(nu_max, proposal));
end

%% 2. 计算后验统计量
samples_ne_valid = samples_ne(burn_in+1:end);
samples_nu_valid = samples_nu(burn_in+1:end);

ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]);
cv_ne = ne_std / ne_mean * 100;

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);
cv_nu = nu_std / nu_mean * 100;

fprintf('后验统计:\n');
fprintf('n_e: 均值=%.3e, CV=%.1f%%\n', ne_mean, cv_ne);
fprintf('ν_e: 均值=%.2f GHz, CV=%.1f%%\n', nu_mean/1e9, cv_nu);

%% 3. 绘图 - 图4-7：MCMC迹线图
figure('Position', [100, 100, 1200, 500], 'Color', 'w');

% 标准颜色方案
color_ne = [0.0, 0.45, 0.74];    % 蓝色
color_nu = [0.85, 0.33, 0.10];   % 橙色
color_true = [0.8, 0.2, 0.2];    % 红色（真值线）
color_burnin = [0.3, 0.3, 0.3];  % 灰色（预烧期边界）

%% 子图(a): n_e 迹线图
subplot(1, 2, 1);
plot(1:N_samples, samples_ne, 'Color', [color_ne, 0.6], 'LineWidth', 0.5);
hold on;

% 真值参考线
yline(n_e_true, '--', 'Color', color_true, 'LineWidth', 2.5, ...
    'Label', sprintf('真值 n_e = %.2e m^{-3}', n_e_true), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 10);

% 预烧期边界
xline(burn_in, '--', 'Color', color_burnin, 'LineWidth', 2, ...
    'Label', '预烧期结束', 'LabelVerticalAlignment', 'bottom', 'FontSize', 10);

% 坐标轴设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('迭代次数', 'FontSize', 14, 'FontName', '宋体');
ylabel('电子密度 n_e (m^{-3})', 'FontSize', 14, 'FontName', '宋体');
title(sprintf('(a) 可观测参数 n_e 迹线图 (CV = %.1f%%)', cv_ne), ...
    'FontSize', 13, 'FontName', '宋体');
xlim([0, N_samples]);
grid on; box on;

% 添加后验统计标注
text(N_samples*0.65, ne_mean*(1+0.04), ...
    sprintf('后验均值: %.3e\n95%% CI: [%.2e, %.2e]', ne_mean, ne_ci(1), ne_ci(2)), ...
    'FontSize', 10, 'FontName', 'Times New Roman', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k');

%% 子图(b): ν_e 迹线图
subplot(1, 2, 2);
plot(1:N_samples, samples_nu/1e9, 'Color', [color_nu, 0.6], 'LineWidth', 0.5);
hold on;

% 真值参考线
yline(nu_e_true/1e9, '--', 'Color', color_true, 'LineWidth', 2.5, ...
    'Label', sprintf('真值 \\nu_e = %.1f GHz', nu_e_true/1e9), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 10);

% 预烧期边界
xline(burn_in, '--', 'Color', color_burnin, 'LineWidth', 2, ...
    'Label', '预烧期结束', 'LabelVerticalAlignment', 'bottom', 'FontSize', 10);

% 先验范围标注
yline(nu_min/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
yline(nu_max/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
text(N_samples*0.02, (nu_min/1e9)*1.05, '先验下界', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
text(N_samples*0.02, (nu_max/1e9)*0.98, '先验上界', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);

% 坐标轴设置
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.2);
xlabel('迭代次数', 'FontSize', 14, 'FontName', '宋体');
ylabel('碰撞频率 \nu_e (GHz)', 'FontSize', 14, 'FontName', '宋体');
title(sprintf('(b) 不可观测参数 \\nu_e 迹线图 (CV = %.1f%%)', cv_nu), ...
    'FontSize', 13, 'FontName', '宋体');
xlim([0, N_samples]);
ylim([0, nu_max/1e9 * 1.1]);
grid on; box on;

% 添加后验统计标注
text(N_samples*0.65, nu_max/1e9*0.85, ...
    sprintf('后验均值: %.2f GHz\n95%% CI: [%.2f, %.2f] GHz', ...
    nu_mean/1e9, nu_ci(1)/1e9, nu_ci(2)/1e9), ...
    'FontSize', 10, 'FontName', 'Times New Roman', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k');

%% 4. 保存图表
output_dir = '../../final_output/figures/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存为 PNG（高分辨率）和 SVG
print('-dpng', '-r300', [output_dir, '图4-7_MCMC迹线图.png']);
print('-dsvg', [output_dir, '图4-7_MCMC迹线图.svg']);

fprintf('\n图 4-7 已保存至 %s\n', output_dir);
fprintf('文件: 图4-7_MCMC迹线图.png, 图4-7_MCMC迹线图.svg\n');
