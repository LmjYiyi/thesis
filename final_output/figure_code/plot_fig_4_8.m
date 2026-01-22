%% plot_fig_4_8.m
% 论文图 4-8：Corner Plot——参数可观测性与耦合结构可视化
% 生成日期：2026-01-22
% 对应章节：4.3.3 参数可观测性判据
%
% 物理意义：
% - 可观测参数（n_e）：边缘后验呈尖锐窄峰（CV < 5%）
% - 不可观测参数（ν_e）：边缘后验呈平坦类均匀分布（CV > 50%）
% - 联合分布呈"纵向长条"结构：n_e被数据约束在窄带内，ν_e在先验范围弥散
%
% 文档描述（第121-123行）：
% "Corner Plot将呈现特征性的'纵向长条'结构：可观测参数在其边缘后验分布上
%  呈现尖锐窄峰（CV < 5%），而不可观测参数呈现平坦的类均匀分布（CV > 50%）"

clear; clc; close all;

%% 1. 加载或模拟MCMC采样数据
% 本代码生成用于论文展示的典型Corner Plot
% 用户可替换为实际LM_MCMC.m运行结果

% 仿真参数设置
N_samples = 10000;
burn_in = 2000;
N_valid = N_samples - burn_in;

% 先验范围
ne_min = 1e18; ne_max = 1e20;
nu_min = 0.1e9; nu_max = 5e9;

% 真值
n_e_true = 4.60e19;     % m^-3
nu_e_true = 1.5e9;      % Hz

% 模拟后验样本
% n_e：窄峰分布（高精度反演）
samples_ne_valid = n_e_true + n_e_true * 0.015 * randn(N_valid, 1);

% ν_e：均匀分布（不可观测）
samples_nu_valid = nu_min + (nu_max - nu_min) * rand(N_valid, 1);

%% 2. 计算后验统计量
ne_mean = mean(samples_ne_valid);
ne_std = std(samples_ne_valid);
ne_ci = prctile(samples_ne_valid, [2.5, 97.5]);
cv_ne = ne_std / ne_mean * 100;

nu_mean = mean(samples_nu_valid);
nu_std = std(samples_nu_valid);
nu_ci = prctile(samples_nu_valid, [2.5, 97.5]);
cv_nu = nu_std / nu_mean * 100;

% 相关系数
rho = corrcoef(samples_ne_valid, samples_nu_valid);
rho_val = rho(1, 2);

fprintf('后验统计:\n');
fprintf('n_e: CV = %.2f%% (可观测)\n', cv_ne);
fprintf('ν_e: CV = %.2f%% (不可观测)\n', cv_nu);
fprintf('参数相关系数 ρ = %.3f\n', rho_val);

%% 3. 绘图 - 图4-8：Corner Plot
figure('Position', [100, 100, 700, 700], 'Color', 'w');

% 标准颜色方案
color_ne = [0.2, 0.6, 0.8];     % 蓝色系
color_nu = [0.8, 0.4, 0.2];     % 橙色系
color_true = [0.8, 0.2, 0.2];   % 红色（真值）
color_scatter = [0.3, 0.5, 0.7];% 散点颜色

%% 子图(1,1): n_e 边缘后验分布
subplot(2, 2, 1);
histogram(samples_ne_valid, 50, 'Normalization', 'pdf', ...
    'FaceColor', color_ne, 'EdgeColor', 'w', 'FaceAlpha', 0.8);
hold on;

% 真值参考线
xline(n_e_true, '--', 'Color', color_true, 'LineWidth', 2.5);

% 95% CI 标注
xline(ne_ci(1), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);
xline(ne_ci(2), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
xlabel('n_e (m^{-3})', 'FontSize', 12);
ylabel('概率密度', 'FontSize', 12, 'FontName', '宋体');
title(sprintf('n_e 边缘后验 (CV = %.1f%%)', cv_ne), 'FontSize', 12, 'FontName', '宋体');

% 添加"尖锐窄峰"标注
text(ne_mean, max(ylim)*0.85, sprintf('\\leftarrow 尖锐窄峰'), ...
    'FontSize', 10, 'Color', color_ne, 'FontWeight', 'bold');

grid on; box on;

%% 子图(2,2): ν_e 边缘后验分布
subplot(2, 2, 4);
histogram(samples_nu_valid/1e9, 50, 'Normalization', 'pdf', ...
    'FaceColor', color_nu, 'EdgeColor', 'w', 'FaceAlpha', 0.8);
hold on;

% 真值参考线
xline(nu_e_true/1e9, '--', 'Color', color_true, 'LineWidth', 2.5);

% 先验范围标注
xline(nu_min/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
xline(nu_max/1e9, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
xlabel('\nu_e (GHz)', 'FontSize', 12);
ylabel('概率密度', 'FontSize', 12, 'FontName', '宋体');
title(sprintf('\\nu_e 边缘后验 (CV = %.1f%%)', cv_nu), 'FontSize', 12, 'FontName', '宋体');

% 添加"平坦分布"标注
text(mean([nu_min, nu_max])/1e9, max(ylim)*0.85, '← 类均匀分布 →', ...
    'FontSize', 10, 'Color', color_nu, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

grid on; box on;

%% 子图(2,1): 联合后验分布（核心：纵向长条结构）
subplot(2, 2, 3);

% 降采样绘制散点图（提高可读性）
subsample_idx = 1:5:N_valid;
scatter(samples_ne_valid(subsample_idx), samples_nu_valid(subsample_idx)/1e9, ...
    8, color_scatter, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

% 真值标注
plot(n_e_true, nu_e_true/1e9, 'r+', 'MarkerSize', 18, 'LineWidth', 3);

% 强调"纵向长条"结构
% 用虚线框标注n_e的约束范围
rectangle('Position', [ne_ci(1), nu_min/1e9, ne_ci(2)-ne_ci(1), (nu_max-nu_min)/1e9], ...
    'EdgeColor', [0.8, 0.2, 0.2], 'LineWidth', 2, 'LineStyle', '--');

% 添加结构标注
annotation('textarrow', [0.28, 0.32], [0.25, 0.35], 'String', '纵向长条结构', ...
    'FontSize', 11, 'FontName', '宋体', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 1.5);

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);
xlabel('n_e (m^{-3})', 'FontSize', 12);
ylabel('\nu_e (GHz)', 'FontSize', 12);
title('联合后验分布', 'FontSize', 12, 'FontName', '宋体');
grid on; box on;

% 标注物理意义
text(ne_mean*0.92, nu_max/1e9*0.9, ...
    sprintf('n_e 被数据约束\n\\nu_e 在先验内弥散'), ...
    'FontSize', 10, 'FontName', '宋体', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'HorizontalAlignment', 'center');

%% 子图(1,2): 参数耦合分析（相关系数）
subplot(2, 2, 2);
axis off;

% 绘制参数耦合信息
text(0.5, 0.7, sprintf('参数相关分析'), ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', '宋体', ...
    'HorizontalAlignment', 'center');

text(0.5, 0.5, sprintf('\\rho_{n_e, \\nu_e} = %.3f', rho_val), ...
    'FontSize', 18, 'FontName', 'Times New Roman', ...
    'HorizontalAlignment', 'center');

if abs(rho_val) < 0.3
    coupling_desc = '弱耦合（可独立反演）';
elseif abs(rho_val) < 0.7
    coupling_desc = '中等耦合';
else
    coupling_desc = '强耦合（需重新参数化）';
end
text(0.5, 0.3, coupling_desc, ...
    'FontSize', 12, 'FontName', '宋体', ...
    'HorizontalAlignment', 'center', 'Color', [0.3, 0.6, 0.3]);

% 添加CV判据说明框
text(0.5, 0.1, sprintf('CV判据：\nn_e: %.1f%% < 5%% → 可观测\n\\nu_e: %.1f%% > 50%% → 不可观测', ...
    cv_ne, cv_nu), ...
    'FontSize', 10, 'FontName', '宋体', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'k', 'Margin', 5);

%% 4. 添加总标题
sgtitle('图 4-8  Drude模型Corner Plot：参数可观测性分析', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', '宋体');

%% 5. 保存图表
output_dir = '../../final_output/figures/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存为 PNG（高分辨率）和 SVG
print('-dpng', '-r300', [output_dir, '图4-8_Corner_Plot.png']);
print('-dsvg', [output_dir, '图4-8_Corner_Plot.svg']);

fprintf('\n图 4-8 已保存至 %s\n', output_dir);
fprintf('文件: 图4-8_Corner_Plot.png, 图4-8_Corner_Plot.svg\n');
