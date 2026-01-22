%% plot_fig_4_3_posterior_comparison.m
% 论文图 4-3：参数可观测性的后验分布对比示意图（尖峰 vs 平原）
% 生成日期：2026-01-22
% 对应章节：4.1.3 反演策略假设：预设ν_e以实现参数降维的可行性论证
%
% 图表描述（摘自定稿文档）：
% "图4-3以示意图形式展示了这一参数可观测性的对比。如图所示，在同一坐标系中
%  绘制了两个概率密度函数：蓝色实线代表n_e的后验分布，呈现出针状的尖锐高斯
%  形态，峰值精准落在真值位置，标准差极小；红色虚线代表ν_e的后验分布，呈现
%  出桌面般的平坦均匀分布，覆盖整个先验范围[0.1, 10] GHz，没有任何向真值
%  聚集的趋势。"

clear; clc; close all;

%% 1. 参数设置
% n_e 后验分布参数（尖锐高斯）
ne_true = 1.04e19;          % 电子密度真值 (m^-3)
ne_std = 0.01 * ne_true;    % 标准差 = 1% (非常尖锐)

% ν_e 后验分布参数（平坦均匀）
nu_prior_min = 0.1e9;       % 先验下界 0.1 GHz
nu_prior_max = 10e9;        % 先验上界 10 GHz
nu_true = 1.5e9;            % 真值 1.5 GHz（但后验不收敛）

%% 2. 创建归一化坐标
% 使用相对坐标便于在同一图中比较

% n_e: 以真值为中心，±5%范围
ne_range_rel = linspace(-5, 5, 500);  % 相对误差 (%)
ne_range_abs = ne_true * (1 + ne_range_rel/100);

% ν_e: 0.1-10 GHz
nu_range = linspace(0.1e9, 10e9, 500);

%% 3. 计算概率密度函数
% n_e 后验：尖锐高斯
pdf_ne = normpdf(ne_range_abs, ne_true, ne_std);
pdf_ne = pdf_ne / max(pdf_ne);  % 归一化到最大值为1

% ν_e 后验：几乎均匀分布（略有起伏模拟噪声）
pdf_nu = ones(size(nu_range)) + 0.05 * randn(size(nu_range));
pdf_nu = abs(pdf_nu);  % 确保非负
pdf_nu = pdf_nu / max(pdf_nu);  % 归一化

%% 4. 绘制对比图
figure('Position', [100, 100, 1000, 500]);

% 左图：n_e 后验分布（尖峰）
subplot(1, 2, 1);
fill([ne_range_rel, fliplr(ne_range_rel)], [pdf_ne, zeros(size(pdf_ne))], ...
     [0.2, 0.5, 0.8], 'EdgeColor', 'b', 'LineWidth', 2, 'FaceAlpha', 0.3);
hold on;
plot(ne_range_rel, pdf_ne, 'b-', 'LineWidth', 2.5);

% 标注真值位置
xline(0, 'k--', 'LineWidth', 1.5);
plot(0, 1, 'b^', 'MarkerSize', 12, 'MarkerFaceColor', 'b');

% 标注标准差范围
xline(-1, 'b:', 'LineWidth', 1);
xline(1, 'b:', 'LineWidth', 1);
text(1.2, 0.6, '\sigma_{n_e} ≈ 1%', 'FontSize', 11, 'FontName', 'SimHei', 'Color', 'b');

hold off;
xlabel('\Delta n_e / n_e^{true} (%)', 'FontSize', 13, 'FontName', 'Times New Roman');
ylabel('归一化后验概率密度', 'FontSize', 13, 'FontName', 'SimHei');
title('(a) n_e 后验分布：尖锐高斯', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
xlim([-5 5]);
ylim([0 1.2]);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

% 添加说明文字
text(-4.5, 1.1, '\bf可观测', 'FontSize', 14, 'FontName', 'SimHei', 'Color', 'b');
text(-4.5, 0.95, '数据提供强约束', 'FontSize', 10, 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);

% 右图：ν_e 后验分布（平原）
subplot(1, 2, 2);
fill([nu_range/1e9, fliplr(nu_range/1e9)], [pdf_nu, zeros(size(pdf_nu))], ...
     [0.8, 0.3, 0.3], 'EdgeColor', 'r', 'LineWidth', 2, 'FaceAlpha', 0.3);
hold on;
plot(nu_range/1e9, pdf_nu, 'r--', 'LineWidth', 2.5);

% 标注真值位置（但后验不收敛）
xline(nu_true/1e9, 'k--', 'LineWidth', 1.5);
plot(nu_true/1e9, pdf_nu(find(nu_range >= nu_true, 1)), 'rv', 'MarkerSize', 12, 'LineWidth', 2);

% 标注先验边界
xline(0.1, 'r:', 'LineWidth', 1.5, 'Label', '先验下界', 'LabelHorizontalAlignment', 'right', 'FontSize', 9);
xline(10, 'r:', 'LineWidth', 1.5, 'Label', '先验上界', 'LabelHorizontalAlignment', 'left', 'FontSize', 9);

hold off;
xlabel('\nu_e (GHz)', 'FontSize', 13, 'FontName', 'Times New Roman');
ylabel('归一化后验概率密度', 'FontSize', 13, 'FontName', 'SimHei');
title('(b) \nu_e 后验分布：平坦均匀', 'FontSize', 14, 'FontName', 'SimHei', 'FontWeight', 'bold');
xlim([0 11]);
ylim([0 1.2]);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2);

% 添加说明文字
text(0.5, 1.1, '\bf不可辨识', 'FontSize', 14, 'FontName', 'SimHei', 'Color', 'r');
text(0.5, 0.95, '数据不提供约束', 'FontSize', 10, 'FontName', 'SimHei', 'Color', [0.3 0.3 0.3]);

% 标注真值与后验均值的偏差
text(nu_true/1e9 + 0.3, 0.5, '真值位置', 'FontSize', 10, 'FontName', 'SimHei', 'Color', 'k');
annotation('arrow', [0.72, 0.68], [0.45, 0.45], 'Color', 'k');

%% 5. 添加总标题
sgtitle('图 4-3 参数可观测性对比：尖峰 vs 平原', 'FontSize', 16, 'FontName', 'SimHei', 'FontWeight', 'bold');

%% 6. 保存图表
print('-dpng', '-r300', '../../final_output/figures/图4-3_后验分布对比_尖峰vs平原.png');
print('-dsvg', '../../final_output/figures/图4-3_后验分布对比_尖峰vs平原.svg');

fprintf('图 4-3 已保存至 final_output/figures/\n');
fprintf('  - (a) n_e 后验分布：尖锐高斯，标准差约1%%\n');
fprintf('  - (b) ν_e 后验分布：平坦均匀，覆盖整个先验范围\n');
fprintf('  - 直观诠释"不可辨识性"的概率定义\n');
