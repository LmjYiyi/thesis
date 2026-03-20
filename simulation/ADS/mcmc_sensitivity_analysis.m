%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 切比雪夫带通滤波器物理参数群时延敏感度分析 (Sensitivity Analysis)
% 目的：可视化中心频率(F0)、绝对带宽(BW)、等效阶数(N)的变动对色散双峰特性的影响规律
% 为 MCMC 贝叶斯反演的收敛能力提供物理学解释
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 基准物理参数设定 (基于目标 ADS 仿真的标称真值)
F0_base = 37.0e9;     % 基准中心频率：37 GHz
BW_base = 1.0e9;      % 基准通带带宽：1 GHz
N_base  = 5;          % 基准等效阶数：5 阶
Ripple  = 0.5;        % 切比雪夫纹波：0.5 dB

f_axis = linspace(35.5e9, 38.5e9, 500); % 宽域扫频观察窗口

fprintf('======================================================\n');
fprintf('  微波物理参数敏感度分析 (Sensitivity Analysis)\n');
fprintf('  基准状态: F0=%.1fGHz, BW=%.1fGHz, N=%d\n', F0_base/1e9, BW_base/1e9, N_base);
fprintf('======================================================\n');

%% 2. 构建三维敏感度对比图阵列
fig_h = figure('Color', 'w', 'Position', [100, 100, 1400, 450]);

% ------------------------------------------------------------------------
% Subplot (a): 对中心频率 F0 的敏感度演变
% 物理规律：双峰结构发生整体刚性平移
% ------------------------------------------------------------------------
subplot(1, 3, 1); hold on; grid on;
F0_vars = [36.5e9, 37.0e9, 37.5e9];
colors_F0 = {[0.2, 0.6, 0.8], [0.8, 0.2, 0.2], [0.2, 0.8, 0.4]};
labels_F0 = {};

for idx = 1:length(F0_vars)
    tau = calculate_chebyshev_group_delay(f_axis, F0_vars(idx), BW_base, N_base, Ripple);
    plot(f_axis/1e9, tau*1e9, 'LineWidth', 1.5, 'Color', colors_F0{idx});
    labels_F0{end+1} = sprintf('F_0 = %.1f GHz', F0_vars(idx)/1e9);
end
legend(labels_F0, 'Location', 'north', 'FontSize', 9, 'NumColumns', 3, 'Box', 'off');
xlabel('探测频率 (GHz)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('解析群延迟 \tau (ns)', 'FontSize', 11, 'FontWeight', 'bold');
title('(a)', 'FontSize', 12);
set(gca, 'FontSize', 10, 'LineWidth', 1.0, 'Box', 'on', 'TickDir', 'in', 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.25); xlim([35.5, 38.5]); ylim([0, 8]);

% ------------------------------------------------------------------------
% Subplot (b): 对绝对带宽 BW 的敏感度演变
% 物理规律：双峰之间的距离发生扩张或收缩，且峰值高度与带宽呈反比 (带宽越窄，谐振越剧烈)
% ------------------------------------------------------------------------
subplot(1, 3, 2); hold on; grid on;
BW_vars = [0.8e9, 1.0e9, 1.2e9];
colors_BW = {[0.2, 0.6, 0.8], [0.8, 0.2, 0.2], [0.2, 0.8, 0.4]};
labels_BW = {};

for idx = 1:length(BW_vars)
    tau = calculate_chebyshev_group_delay(f_axis, F0_base, BW_vars(idx), N_base, Ripple);
    plot(f_axis/1e9, tau*1e9, 'LineWidth', 1.5, 'Color', colors_BW{idx});
    labels_BW{end+1} = sprintf('BW = %.1f GHz', BW_vars(idx)/1e9);
end
legend(labels_BW, 'Location', 'north', 'FontSize', 9, 'NumColumns', 3, 'Box', 'off');
xlabel('探测频率 (GHz)', 'FontSize', 11, 'FontWeight', 'bold');
% ylabel('解析群延迟 \tau (ns)', 'FontSize', 11, 'FontWeight', 'bold');
title('(b)', 'FontSize', 12);
set(gca, 'FontSize', 10, 'LineWidth', 1.0, 'Box', 'on', 'TickDir', 'in', 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.25); xlim([35.5, 38.5]); ylim([0, 8]);

% ------------------------------------------------------------------------
% Subplot (c): 对滤波器等效阶数 N 的敏感度演变
% 物理规律：阶数越高，通带内部越不平坦（微峰变多），且边缘谐振截断越尖锐、极值越高
% ------------------------------------------------------------------------
subplot(1, 3, 3); hold on; grid on;
N_vars = [3, 5, 7];
colors_N = {[0.2, 0.6, 0.8], [0.8, 0.2, 0.2], [0.2, 0.8, 0.4]};
labels_N = {};

for idx = 1:length(N_vars)
    tau = calculate_chebyshev_group_delay(f_axis, F0_base, BW_base, N_vars(idx), Ripple);
    plot(f_axis/1e9, tau*1e9, 'LineWidth', 1.5, 'Color', colors_N{idx});
    labels_N{end+1} = sprintf('阶数 N = %d', N_vars(idx));
end
legend(labels_N, 'Location', 'north', 'FontSize', 9, 'NumColumns', 3, 'Box', 'off');
xlabel('探测频率 (GHz)', 'FontSize', 11, 'FontWeight', 'bold');
% ylabel('解析群延迟 \tau (ns)', 'FontSize', 11, 'FontWeight', 'bold');
title('(c)', 'FontSize', 12);
set(gca, 'FontSize', 10, 'LineWidth', 1.0, 'Box', 'on', 'TickDir', 'in', 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.25); xlim([35.5, 38.5]); ylim([0, 8]);

% Export figure
export_thesis_figure(fig_h, '图5-敏感度分析', 14, 600);

disp('所有敏感度曲线绘制完成。');

%% ================= 本地切比雪夫连续传递模型计算模块 =================
function tau_g = calculate_chebyshev_group_delay(f_vec, F0, BW, N, Ripple)
    % 严格模拟切比雪夫带通滤波器群延迟物理模型
    % 基于真实的传递函数计算相导数，完美重构色散双峰与多峰谐振
    
    N_int = round(N);
    if N_int < 1, N_int = 1; end
    
    W1 = 2 * pi * (F0 - BW/2);
    W2 = 2 * pi * (F0 + BW/2);
    
    if W1 >= W2
        tau_g = zeros(size(f_vec));
        return;
    end
    
    try
        [b, a] = cheby1(N_int, Ripple, [W1, W2], 'bandpass', 's');
        w_vec = 2 * pi * f_vec;
        H = freqs(b, a, w_vec);
        
        phase = unwrap(angle(H));
        tau_g = -gradient(phase) ./ gradient(w_vec);
        tau_g(tau_g < 0) = 0; % 清除伪影
    catch
        tau_g = zeros(size(f_vec));
    end
end

%% ================= 论文图像导出函数 =================
function export_thesis_figure(fig_handle, out_name, width_cm, dpi)
% 简化版导出函数：保持原有字体渲染，仅格式化布局
if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 600; end

height_cm = width_cm * 0.65;
out_dir = fullfile(pwd, 'figures_export');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPositionMode', 'auto', ...
    'PaperSize', [width_cm, height_cm]);

% 隐藏 sgtitle
title_nodes = [ ...
    findall(fig_handle, 'Type', 'Text', 'Tag', 'suptitle'); ...
    findall(fig_handle, 'Type', 'Text', 'Tag', 'sgtitle') ...
];
if ~isempty(title_nodes)
    set(title_nodes, 'Visible', 'off');
end

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
file_emf = fullfile(out_dir, [out_name, '.emf']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi, 'BackgroundColor', 'white');
try
    exportgraphics(fig_handle, file_emf, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch
    warning('EMF export failed on current platform.');
end
fprintf('[export] %s\n', file_tiff);
fprintf('[export] %s\n', file_emf);
end