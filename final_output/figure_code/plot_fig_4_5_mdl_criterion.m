%% plot_fig_4_5_mdl_criterion_thesis.m
% 论文图 4-5：MDL准则性能验证（博士论文终稿版）
% 说明：
% 1) 图题不放在图内，由论文正文题注给出
% 2) 输出 TIFF(600 dpi) + PDF(矢量) + EMF(若平台支持)
% 3) 不人为调整AIC结果，保持仿真结果可追溯

clear; clc; close all;
rng(20260122);

fprintf('===== 图4-5: MDL准则性能验证（论文终稿版） =====\n');

%% 0. 绘图/导出参数
cn_font   = 'SimSun';              % 中文字体；如异常可改为 'Microsoft YaHei'
en_font   = 'Times New Roman';     % 英文字体
font_ax   = 10.5;                  % 坐标轴刻度字号
font_lab  = 11;                    % 坐标轴标签字号
font_leg  = 10;                    % 图例字号
font_anno = 11;                    % 子图角标字号
font_note = 9.5;                   % 轻量注释字号

fig_width_cm  = 16.0;              % 双子图并排宽度
fig_height_cm = 7.4;               % 高度
dpi_out       = 600;

%% 1. 参数设置
f_s = 1e6;                  % 采样率 (Hz)
N_samples = 48;             % 窗口内采样点数
L = 24;                     % 子空间维度 (L = N/2)
M = N_samples - L + 1;      % 快拍数

% 真实信源参数
K_true = 2;                 % 真实信源数
f_sources = [150e3, 320e3]; % 两个信源频率 (Hz)
A_sources = [1.0, 0.5];     % 信源幅度

% SNR扫描范围
SNR_list = [30, 20, 10];    % 建议高到低排列，更符合阅读习惯
k_candidates = 0:5;         % 候选信源数

% 蒙特卡洛仿真次数
N_MC = 100;

%% 2. 蒙特卡洛仿真
MDL_results = zeros(length(SNR_list), length(k_candidates));
AIC_results = zeros(length(SNR_list), length(k_candidates));
MDL_std = zeros(length(SNR_list), length(k_candidates));

for snr_idx = 1:length(SNR_list)
    SNR_dB = SNR_list(snr_idx);

    mdl_mc = zeros(N_MC, length(k_candidates));
    aic_mc = zeros(N_MC, length(k_candidates));

    for mc = 1:N_MC
        % 生成含噪多信源信号
        n = (0:N_samples-1)';
        signal = zeros(N_samples, 1);
        for s = 1:K_true
            phase = 2*pi*rand;
            signal = signal + A_sources(s) * exp(1i * (2*pi*f_sources(s)/f_s*n + phase));
        end

        % 添加噪声
        noise_power = 10^(-SNR_dB/10) * mean(abs(signal).^2);
        noise = sqrt(noise_power/2) * (randn(N_samples,1) + 1i*randn(N_samples,1));
        x = signal + noise;

        % 构造 Hankel 矩阵
        X_hankel = zeros(L, M);
        for col = 1:M
            X_hankel(:, col) = x(col : col+L-1);
        end

        % 前后向平均协方差矩阵
        R_f = (X_hankel * X_hankel') / M;
        J_mat = fliplr(eye(L));
        R_x = (R_f + J_mat * conj(R_f) * J_mat) / 2;

        % 特征值分解
        lambda = sort(real(eig(R_x)), 'descend');

        % 计算各候选k的MDL和AIC
        for k_idx = 1:length(k_candidates)
            k = k_candidates(k_idx);
            mdl_mc(mc, k_idx) = compute_mdl(lambda, k, L, M);
            aic_mc(mc, k_idx) = compute_aic(lambda, k, L, M);
        end
    end

    MDL_results(snr_idx, :) = mean(mdl_mc, 1);
    AIC_results(snr_idx, :) = mean(aic_mc, 1);
    MDL_std(snr_idx, :) = std(mdl_mc, 0, 1);
end

%% 3. 归一化（便于同图对比）
for i = 1:length(SNR_list)
    MDL_results(i, :) = MDL_results(i, :) - min(MDL_results(i, :));
    AIC_results(i, :) = AIC_results(i, :) - min(AIC_results(i, :));
end

%% 4. 论文终稿绘图
fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [2, 2, fig_width_cm, fig_height_cm], ...
             'PaperUnits', 'centimeters', ...
             'PaperPositionMode', 'auto', ...
             'PaperSize', [fig_width_cm, fig_height_cm]);

tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 克制配色
colors = [
    0.00, 0.32, 0.74;   % 30 dB - 蓝
    0.85, 0.33, 0.10;   % 20 dB - 橙
    0.49, 0.18, 0.56;   % 10 dB - 紫
];
markers = {'o', 's', '^'};

%% 子图 (a): MDL 在不同 SNR 下的结果
ax1 = nexttile(tl, 1);
hold(ax1, 'on');

h_mdl = gobjects(length(SNR_list), 1);
for snr_idx = 1:length(SNR_list)
    h_mdl(snr_idx) = plot(ax1, k_candidates, MDL_results(snr_idx, :), ...
        '-', ...
        'Color', colors(snr_idx, :), ...
        'LineWidth', 1.5, ...
        'Marker', markers{snr_idx}, ...
        'MarkerSize', 6.5, ...
        'MarkerFaceColor', colors(snr_idx, :), ...
        'DisplayName', sprintf('SNR = %d dB', SNR_list(snr_idx)));
end

% 标出真实信源数
xline(ax1, K_true, ':', ...
    'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.0, ...
    'HandleVisibility', 'off');

% 轻量标注
yl1 = ylim(ax1);
text(ax1, K_true + 0.08, yl1(1) + 0.88*(yl1(2)-yl1(1)), ...
    '\fontname{Times New Roman}K_{true}=2', ...
    'Interpreter', 'tex', ...
    'FontName', en_font, ...
    'FontSize', font_note, ...
    'Color', [0.15 0.15 0.15]);

format_axes(ax1, en_font, font_ax);

xlabel(ax1, '\fontname{SimSun}候选信源数 \fontname{Times New Roman}k', ...
    'Interpreter', 'tex', 'FontSize', font_lab);
ylabel(ax1, '\fontname{Times New Roman}MDL \fontname{SimSun}代价函数值（归一化）', ...
    'Interpreter', 'tex', 'FontSize', font_lab);

xlim(ax1, [-0.5, 5.5]);
xticks(ax1, 0:5);

lg1 = legend(ax1, h_mdl, ...
    arrayfun(@(x) sprintf('SNR = %d dB', x), SNR_list, 'UniformOutput', false), ...
    'Location', 'northeast');
format_legend(lg1, cn_font, font_leg);

text(ax1, 0.02, 0.96, '(a)', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontName', en_font, ...
    'FontSize', font_anno, ...
    'FontWeight', 'bold');

%% 子图 (b): MDL vs AIC 对比（低 SNR）
ax2 = nexttile(tl, 2);
hold(ax2, 'on');

% 取最低SNR=10 dB，对应第3条
idx_low_snr = find(SNR_list == 10, 1);

h1 = plot(ax2, k_candidates, MDL_results(idx_low_snr, :), ...
    '-', ...
    'Color', [0.00, 0.32, 0.74], ...
    'LineWidth', 1.5, ...
    'Marker', 'o', ...
    'MarkerSize', 6.5, ...
    'MarkerFaceColor', [0.00, 0.32, 0.74], ...
    'DisplayName', 'MDL');

h2 = plot(ax2, k_candidates, AIC_results(idx_low_snr, :), ...
    '--', ...
    'Color', [0.85, 0.33, 0.10], ...
    'LineWidth', 1.5, ...
    'Marker', 's', ...
    'MarkerSize', 6.5, ...
    'MarkerFaceColor', [0.85, 0.33, 0.10], ...
    'DisplayName', 'AIC');

xline(ax2, K_true, ':', ...
    'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.0, ...
    'HandleVisibility', 'off');

yl2 = ylim(ax2);
text(ax2, K_true + 0.08, yl2(1) + 0.88*(yl2(2)-yl2(1)), ...
    '\fontname{Times New Roman}K_{true}=2', ...
    'Interpreter', 'tex', ...
    'FontName', en_font, ...
    'FontSize', font_note, ...
    'Color', [0.15 0.15 0.15]);

% 若AIC最小值出现在过估计位置，则做轻量说明
[~, aic_min_idx] = min(AIC_results(idx_low_snr, :));
k_aic_min = k_candidates(aic_min_idx);
if k_aic_min > K_true
    text(ax2, k_aic_min + 0.08, AIC_results(idx_low_snr, aic_min_idx) + 0.08*range(AIC_results(idx_low_snr, :)), ...
        'AIC偏向过估计', ...
        'FontName', cn_font, ...
        'FontSize', 9, ...
        'Color', [0.45 0.20 0.10]);
end

format_axes(ax2, en_font, font_ax);

xlabel(ax2, '\fontname{SimSun}候选信源数 \fontname{Times New Roman}k', ...
    'Interpreter', 'tex', 'FontSize', font_lab);
ylabel(ax2, '\fontname{SimSun}代价函数值（归一化）', ...
    'Interpreter', 'tex', 'FontSize', font_lab);

xlim(ax2, [-0.5, 5.5]);
xticks(ax2, 0:5);

lg2 = legend(ax2, [h1, h2], {'MDL', 'AIC'}, ...
    'Location', 'northeast');
format_legend(lg2, cn_font, font_leg);

text(ax2, 0.02, 0.96, '(b)', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontName', en_font, ...
    'FontSize', font_anno, ...
    'FontWeight', 'bold');

%% 5. 导出图表
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir, 'figures_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_name_base = '图4-5_MDL准则性能验证';

export_thesis_figure(fig, output_dir, fig_name_base, dpi_out);

fprintf('\n✓ 图 4-5 已导出\n');
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.tiff']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.pdf']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.emf']));

%% ===== 辅助函数 =====
function mdl = compute_mdl(lambda, k, p, N_snaps)
    if k >= p
        mdl = Inf;
        return;
    end

    noise_evals = lambda(k+1:end);
    noise_evals(noise_evals < 1e-15) = 1e-15;

    g_mean = prod(noise_evals)^(1/length(noise_evals));
    a_mean = mean(noise_evals);

    term1 = -(p-k) * N_snaps * log(g_mean / a_mean);
    term2 = 0.5 * k * (2*p - k) * log(N_snaps);
    mdl = term1 + term2;
end

function aic = compute_aic(lambda, k, p, N_snaps)
    if k >= p
        aic = Inf;
        return;
    end

    noise_evals = lambda(k+1:end);
    noise_evals(noise_evals < 1e-15) = 1e-15;

    g_mean = prod(noise_evals)^(1/length(noise_evals));
    a_mean = mean(noise_evals);

    term1 = -(p-k) * N_snaps * log(g_mean / a_mean);
    term2 = k * (2*p - k);
    aic = term1 + term2;
end

function format_axes(ax, en_font, font_ax)
    set(ax, ...
        'FontName', en_font, ...
        'FontSize', font_ax, ...
        'LineWidth', 0.9, ...
        'Box', 'on', ...
        'TickDir', 'in', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.18, ...
        'GridLineStyle', '-');
end

function format_legend(lg, cn_font, font_leg)
    set(lg, ...
        'FontName', cn_font, ...
        'FontSize', font_leg, ...
        'Interpreter', 'tex', ...
        'Box', 'on', ...
        'Color', 'white', ...
        'EdgeColor', [0.7 0.7 0.7], ...
        'LineWidth', 0.6, ...
        'AutoUpdate', 'off');
end

function export_thesis_figure(fig_handle, out_dir, out_name, dpi)
    set(fig_handle, 'Color', 'w');

    file_tiff = fullfile(out_dir, [out_name, '.tiff']);
    file_pdf  = fullfile(out_dir, [out_name, '.pdf']);
    file_emf  = fullfile(out_dir, [out_name, '.emf']);

    exportgraphics(fig_handle, file_tiff, ...
        'Resolution', dpi, ...
        'BackgroundColor', 'white');

    exportgraphics(fig_handle, file_pdf, ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    try
        exportgraphics(fig_handle, file_emf, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    catch
        warning('EMF export failed on current platform.');
    end
end