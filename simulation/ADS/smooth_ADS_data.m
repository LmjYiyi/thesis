% @[simulation/ADS/smooth_ADS_data.m]
% Script to process and smooth ADS spectrum data for visualization
% Constraints:
% - Preserve signal bandwidth
% - Preserve passband edge frequencies
% - Preserve in-band average power
% - Preserve peak position and -3 dB bandwidth

clear; clc; close all;

% 全局字体设置（解决中文显示，带回退，参考用户现有代码）
font_cn = 'SimSun';
try
    listfonts(font_cn);
catch
    font_cn = 'Microsoft YaHei';
end
set(groot, 'defaultTextFontName', font_cn);
set(groot, 'defaultAxesFontName', font_cn);

%% 1. Load Data
% Check if file exists
file_path = 'fashe_dbm.txt';
if ~isfile(file_path)
    error('File %s not found in current directory.', file_path);
end

% Read data (Skip 1 header line)
% Format is assumed to be: freq <tab/space> dBm
try
    data = readmatrix(file_path, 'FileType', 'text', 'NumHeaderLines', 1);
catch
    % Fallback for older MATLAB versions or different formats
    raw = importdata(file_path);
    data = raw.data;
end

freq = data(:, 1);       % Hz
power_dBm = data(:, 2);  % dBm

% Calculate frequency resolution
dfs = diff(freq);
df = mean(dfs);
fprintf('数据已加载。点数：%d\n', length(freq));
fprintf('平均频率步进：%.2f kHz\n', df/1e3);

%% 2. Processing Steps

% --- Step 1: Median Filter (Remove Impulsive Outliers) ---
% Use a small window (e.g., 3-5 points) to remove single-point spikes
med_window = 3; 
power_med = medfilt1(power_dBm, med_window, 'truncate');

% --- Step 2: Equivalent RBW Smoothing (Frequency Binning) ---
% Approach: Moving Average on Linear Power (Energy Preservation)
% This creates a "sliding window" equivalent to an RBW filter.
% Using linear power ensures "In-band Average Power" is preserved correcty.

RBW_target = 60e6; % 60 MHz (Configurable)
window_len = round(RBW_target / df);

% Ensure window length is odd for symmetry (helps preserve peak position)
if mod(window_len, 2) == 0
    window_len = window_len + 1;
end

fprintf('正在应用等效 RBW 平滑：%.1f MHz (窗口：%d 点)\n', ...
    RBW_target/1e6, window_len);

% Convert to Linear Power (mW)
power_mW = 10.^(power_med ./ 10);

% Apply Moving Average (Boxcar Filter)
% 'movmean' computes the centered moving average
power_smooth_mW = movmean(power_mW, window_len, 'Endpoints', 'shrink');

% Convert back to dBm
power_smooth_dBm = 10 .* log10(power_smooth_mW);

% --- Step 3: Optional Mild Savitzky-Golay Smoothing ---
% Used strictly for visualization smoothness on the dBm curve.
% Parameters: Order 2 (less aggressive), Frame Length 11 (preserves sharp features).
sg_order = 2;
sg_len = 51; % Stronger smoothing length for ripples

% Adjust sg_len if data is too short
if sg_len > length(power_smooth_dBm)
    sg_len = length(power_smooth_dBm);
    if mod(sg_len,2)==0, sg_len=sg_len-1; end
end

% Apply SG filter
power_final = sgolayfilt(power_smooth_dBm, sg_order, sg_len);

%% 3. 定量分析：-3 dB 带宽验证
% 目的：计算并验证 -3 dB 带宽指标

% 1. 计算 -3 dB 带宽
[max_val_final, max_idx_final] = max(power_final);
target_level_3dB = max_val_final - 3;
idx_left_3 = find(power_final(1:max_idx_final) <= target_level_3dB, 1, 'last');
% --- 修复后的右边沿查找逻辑 ---
% 找到峰值右侧【最后一次】大于等于 -3dB 的相对位置
last_above_3dB_offset = find(power_final(max_idx_final:end) >= target_level_3dB, 1, 'last');

% 它的下一个点就是真正的右侧跌落点
if isempty(last_above_3dB_offset)
    idx_right_3 = [];
else
    idx_right_3 = last_above_3dB_offset + max_idx_final;
    
    % 防止超出数组边界的安全机制
    if idx_right_3 > length(power_final)
        idx_right_3 = length(power_final);
    end
end

if ~isempty(idx_left_3) && ~isempty(idx_right_3)
    % 线性插值优化精度
    f_left_3 = freq(idx_left_3) + (target_level_3dB - power_final(idx_left_3)) * (freq(idx_left_3+1) - freq(idx_left_3)) / (power_final(idx_left_3+1) - power_final(idx_left_3));
    f_right_3 = freq(idx_right_3-1) + (target_level_3dB - power_final(idx_right_3-1)) * (freq(idx_right_3) - freq(idx_right_3-1)) / (power_final(idx_right_3) - power_final(idx_right_3-1));
    bw_3dB = f_right_3 - f_left_3;
else
    bw_3dB = NaN;
end

fprintf('\n--- 最终验证结论 ---\n');
if ~isnan(bw_3dB)
    fprintf('-3 dB 带宽:   %.2f GHz (%.2f MHz)\n', bw_3dB/1e9, bw_3dB/1e6);
    fprintf('   左边沿:    %.4f GHz\n', f_left_3/1e9);
    fprintf('   右边沿:    %.4f GHz\n', f_right_3/1e9);
else
    fprintf('-3 dB 带宽计算失败 (未找到交叉点)\n');
end
fprintf('--------------------------\n');

%% 4. 可视化 (Visualization) - 严格展示 -3dB 带宽
figure('Name', 'ADS Spectrum Analysis (-3dB)', 'Color', 'w', 'Position', [100, 100, 900, 600]);
hold on; grid on; box on;

% 画平滑后的谱
% 使用深蓝色加粗线，突出主信号
h1 = plot(freq/1e9, power_final, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2.0, 'DisplayName', '平滑频谱');

% 标注 -3 dB (重点突出，黑色虚线)
if ~isnan(bw_3dB)
    yline(max_val_final - 3, '--k', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('-3 dB BW = %.2f GHz', bw_3dB/1e9));
    
    % 垂直辅助线 (可选，增加清晰度)
    xline(f_left_3/1e9, ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    xline(f_right_3/1e9, ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', 1.0, 'HandleVisibility', 'off');

    % 在上方添加文字标注框
    text_x = (f_left_3 + f_right_3)/2 / 1e9;
    text_y = max_val_final + 3;  % 在峰值上方 3 dB 处悬浮显示
    text(text_x, text_y, sprintf('-3 dB 带宽 = %.2f GHz', bw_3dB/1e9), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontName', font_cn, ...
        'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 5);
end

xlabel('频率 (GHz)', 'FontSize', 12, 'FontName', font_cn, 'FontWeight', 'bold');
ylabel('功率 (dBm)', 'FontSize', 12, 'FontName', font_cn, 'FontWeight', 'bold');
title('LFMCW 频谱带宽验证 (-3 dB)', 'FontSize', 14, 'FontName', font_cn);

% 图例
legend('Location', 'NorthEast', 'FontSize', 10, 'FontName', font_cn);

% 坐标轴美化
set(gca, 'FontSize', 11, 'FontName', font_cn, 'LineWidth', 1.0);
axis tight; 
% 扩展 Y 轴范围，为上方文字标注留出空间
yl = ylim;
ylim([yl(1)-2, yl(2)+8]); 

% Optional: Save plot
% export_filename = 'smoothed_spectrum_plot_final_3dB';
% print(gcf, [export_filename '.png'], '-dpng', '-r300');
% fprintf('图像已保存为 %s.png\n', export_filename);

hold off;

% --- 辅助函数：画双向箭头 ---
function draw_arrow(x1, x2, y, color, text_str)
    % 画横线
    line([x1, x2], [y, y], 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % 画左右箭头 (简单的 'v' 字形旋转)
    arrow_size_x = (x2-x1)*0.02; % 箭头宽度比例
    arrow_size_y = 1.5;          % 箭头高度 (dB)
    
    % 左箭头
    line([x1, x1+arrow_size_x], [y, y+arrow_size_y], 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    line([x1, x1+arrow_size_x], [y, y-arrow_size_y], 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % 右箭头
    line([x2, x2-arrow_size_x], [y, y+arrow_size_y], 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    line([x2, x2-arrow_size_x], [y, y-arrow_size_y], 'Color', color, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % 居中文字标注
    text((x1+x2)/2, y-2.5, text_str, 'Color', color, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
end
