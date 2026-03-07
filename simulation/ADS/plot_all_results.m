% @[simulation/ADS/plot_all_results.m]
% Script to visualize ADS simulation results: S21, Delay, and Joint Time-Frequency Analysis
% Matches code style of smooth_ADS_data.m

clear; clc; close all;

% --- 0. Font Setup (Chinese Support) ---
font_cn = 'SimSun';
try
    listfonts(font_cn);
catch
    font_cn = 'Microsoft YaHei';
end
set(groot, 'defaultTextFontName', font_cn);
set(groot, 'defaultAxesFontName', font_cn);

% --- 1. File Definitions ---
% Define all files to be plotted in groups
files_spectrum = {
    'fashe_dbm.txt',   '发射信号频谱 (Fashe Spectrum)';
    'jieshou_dbm.txt', '接收信号频谱 (Jieshou Spectrum)';
    'hunpin_dbm.txt',  '混频信号频谱 (Hunpin Spectrum)'
};

files_time = {
    'fashe_time_v.txt',   '发射信号时域 (Fashe Time)';
    'jieshou_time_v.txt', '接收信号时域 (Jieshou Time)';
    'hunpin_time_v.txt',  '混频信号时域 (Hunpin Time)'
};

% files_sparam removed - now using individual figures

%% --- 3. Plot S21 (Figure 1) ---
fig1 = figure('Name', 'ADS S21 Analysis', 'Color', 'w', 'Position', [100, 100, 1200, 500]);

filename = 's21.txt';
title_str = 'S21 幅值 (dB)';
[freq, data, ok] = load_ads_data(filename);

if ok
    subplot(1, 1, 1);
    ylabel('幅值 (dB)', 'FontSize', 11, 'FontName', font_cn);

    if max(freq) >= 1e9
        plot(freq/1e9, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
        xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', font_cn);
    else
        plot(freq/1e6, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
        xlabel('频率 (MHz)', 'FontSize', 11, 'FontName', font_cn);
    end

    title(title_str, 'FontSize', 13, 'FontName', font_cn, 'FontWeight', 'bold');
    grid on;
    % 给顶部留10%空白
    ylim([min(data) - range(data)*0.1, max(data) + range(data)*0.1]);
end

% 导出论文格式图像
export_thesis_figure(fig1, 'ads_s21', 14, 600, font_cn);

%% --- 4. Plot Delay (Figure 2) ---
fig2 = figure('Name', 'ADS Delay Analysis', 'Color', 'w', 'Position', [150, 150, 1200, 500]);

filename = 'delay.txt';
title_str = '群延迟 (Delay)';
[freq, data, ok] = load_ads_data(filename);

if ok
    subplot(1, 1, 1);
    ylabel('延迟 (ns)', 'FontSize', 11, 'FontName', font_cn);
    data = data * 1e9;

    if max(freq) >= 1e9
        plot(freq/1e9, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
        xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', font_cn);
    else
        plot(freq/1e6, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
        xlabel('频率 (MHz)', 'FontSize', 11, 'FontName', font_cn);
    end

    title(title_str, 'FontSize', 13, 'FontName', font_cn, 'FontWeight', 'bold');
    grid on;
    % 给顶部留10%空白
    ylim([min(data) - range(data)*0.1, max(data) + range(data)*0.1]);
end

% 导出论文格式图像
export_thesis_figure(fig2, 'ads_delay', 14, 600, font_cn);

%% --- 5. 综合时频域分析 (Figure 3: 2x3 Layout) ---
fig3 = figure('Name', 'ADS 时频域联合分析', 'Color', 'w', 'Position', [100, 100, 1400, 800]);

signal_labels = {'发射信号 (TX)', '接收信号 (RX)', '混频信号 (IF)'};

for i = 1:3
    % --- Top Subplot: Time Domain ---
    filename_time = files_time{i, 1};
    title_time = files_time{i, 2};
    [t, v, ok_t] = load_ads_data(filename_time);

    if ok_t
        subplot(2, 3, i);
        % Auto-scale time unit
        if max(t) < 1e-9
            t_plot = t * 1e12;
            x_label_str = '时间 (ps)';
        elseif max(t) < 1e-6
            t_plot = t * 1e9;
            x_label_str = '时间 (ns)';
        elseif max(t) < 1e-3
            t_plot = t * 1e6;
            x_label_str = '时间 (us)';
        else
            t_plot = t;
            x_label_str = '时间 (s)';
        end

        plot(t_plot, v, 'LineWidth', 1.2, 'Color', [0.8500, 0.3250, 0.0980]);
        xlabel(x_label_str, 'FontSize', 10, 'FontName', font_cn);
        ylabel('电压 (V)', 'FontSize', 10, 'FontName', font_cn);
        title(sprintf('%s - 时域', signal_labels{i}), 'FontSize', 11, 'FontName', font_cn, 'FontWeight', 'bold');
        grid on; axis tight;
    end

    % --- Bottom Subplot: Spectrum ---
    filename_spec = files_spectrum{i, 1};
    title_spec = files_spectrum{i, 2};
    [freq, dbm, ok_f] = load_ads_data(filename_spec);

    if ok_f
        dbm = smooth_spectrum(freq, dbm);
        subplot(2, 3, i + 3);
        % Auto-scale frequency unit
        if max(freq) >= 1e9
            plot(freq/1e9, dbm, 'LineWidth', 1.2, 'Color', [0, 0.4470, 0.7410]);
            xlabel('频率 (GHz)', 'FontSize', 10, 'FontName', font_cn);
        elseif max(freq) >= 1e6
            plot(freq/1e6, dbm, 'LineWidth', 1.2, 'Color', [0, 0.4470, 0.7410]);
            xlabel('频率 (MHz)', 'FontSize', 10, 'FontName', font_cn);
        else
            plot(freq, dbm, 'LineWidth', 1.2, 'Color', [0, 0.4470, 0.7410]);
            xlabel('频率 (Hz)', 'FontSize', 10, 'FontName', font_cn);
        end

        ylabel('功率 (dBm)', 'FontSize', 10, 'FontName', font_cn);
        title(sprintf('%s - 频谱', signal_labels{i}), 'FontSize', 11, 'FontName', font_cn, 'FontWeight', 'bold');
        grid on;
        % 给顶部留10%空白，避免频谱贴着最上面
        ylim([min(dbm) - range(dbm)*0.1, max(dbm) + range(dbm)*0.1]);
    end
end

% 导出论文格式图像 (双列通栏图，14cm宽)
export_thesis_figure(fig3, 'ads_time_freq_joint', 14, 600, font_cn);

fprintf('所有绘图已完成。\n');

% --- 6. Helper Function to Load Data ---
function [x, y, success] = load_ads_data(filename)
    if ~isfile(filename)
        warning('File %s not found.', filename);
        x = []; y = []; success = false;
        return;
    end
    try
        data = readmatrix(filename, 'FileType', 'text', 'NumHeaderLines', 1);
        x = data(:, 1);
        y = data(:, 2);
        success = true;
    catch
        warning('Failed to read %s.', filename);
        x = []; y = []; success = false;
    end
end

% --- 7. Helper Function to Smooth Spectrum ---
function [power_final] = smooth_spectrum(freq, power_dBm)
    if length(freq) < 10
        power_final = power_dBm;
        return;
    end
    
    dfs = diff(freq);
    df = mean(dfs);
    if df == 0 || isnan(df)
        power_final = power_dBm;
        return;
    end
    
    % Step 1: Median Filter
    med_window = 3; 
    power_med = medfilt1(power_dBm, med_window, 'truncate');
    
    % Step 2: Equivalent RBW Smoothing (60 MHz)
    RBW_target = 60e6; 
    window_len = round(RBW_target / df);
    if isnan(window_len) || window_len <= 0
        window_len = 1;
    end
    if mod(window_len, 2) == 0
        window_len = window_len + 1;
    end
    
    power_mW = 10.^(power_med ./ 10);
    power_smooth_mW = movmean(power_mW, window_len, 'Endpoints', 'shrink');
    power_smooth_dBm = 10 .* log10(power_smooth_mW);
    
    % Step 3: SG Smoothing
    sg_order = 2;
    sg_len = 51; 
    if sg_len > length(power_smooth_dBm)
        sg_len = length(power_smooth_dBm);
        if mod(sg_len,2)==0, sg_len=sg_len-1; end
    end
    
    if sg_len >= 3 && length(power_smooth_dBm) >= sg_len
        power_final = sgolayfilt(power_smooth_dBm, sg_order, sg_len);
    else
        power_final = power_smooth_dBm;
    end
end

% --- 8. Export Thesis Figure Function ---
function export_thesis_figure(fig_handle, out_name, width_cm, dpi, cn_font)
% Standardize thesis figure style and export TIFF/EMF.

if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 600; end
if nargin < 5 || isempty(cn_font), cn_font = 'SimSun'; end

en_font = 'Times New Roman';
height_cm = width_cm * 0.65;

out_dir = 'figures_export';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPositionMode', 'auto', ...
    'PaperSize', [width_cm, height_cm]);

hide_sgtitle(fig_handle);
style_axes(fig_handle, en_font, cn_font);
style_lines(fig_handle);
style_legend(fig_handle, cn_font, en_font);

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

function hide_sgtitle(fig_handle)
title_nodes = findall(fig_handle, 'Type', 'Text', 'Tag', 'suptitle');
if ~isempty(title_nodes)
    set(title_nodes, 'Visible', 'off');
end
end

function style_axes(fig_handle, en_font, cn_font)
ax_all = findall(fig_handle, 'Type', 'axes');
for i = 1:numel(ax_all)
    ax = ax_all(i);
    if isprop(ax, 'Tag') && strcmpi(ax.Tag, 'legend')
        continue;
    end

    set(ax, ...
        'FontName', en_font, ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.25, ...
        'TickDir', 'in');

    xl = ax.XLabel;
    if ~isempty(xl) && ~isempty(xl.String)
        set(xl, 'FontName', cn_font, 'FontSize', 10, 'Interpreter', 'tex');
    end

    for j = 1:numel(ax.YAxis)
        yl = ax.YAxis(j).Label;
        if ~isempty(yl) && ~isempty(yl.String)
            set(yl, 'FontName', cn_font, 'FontSize', 10, 'Interpreter', 'tex');
        end
    end

    tl = ax.Title;
    if ~isempty(tl) && ~isempty(tl.String)
        set(tl, 'FontName', cn_font, 'FontSize', 10, 'FontWeight', 'normal');
    end
end
end

function style_lines(fig_handle)
line_all = findall(fig_handle, 'Type', 'line');
for i = 1:numel(line_all)
    if strcmp(get(line_all(i), 'LineStyle'), 'none')
        set(line_all(i), 'LineWidth', 1.0);
    else
        set(line_all(i), 'LineWidth', 1.5);
    end
end
end

function style_legend(fig_handle, cn_font, ~)
legend_all = findall(fig_handle, 'Type', 'legend');
for i = 1:numel(legend_all)
    lg = legend_all(i);

    set(lg, ...
        'FontName', cn_font, ...
        'FontSize', 10, ...
        'Box', 'off', ...
        'Color', 'none', ...
        'Interpreter', 'tex', ...
        'AutoUpdate', 'off');

    if strcmpi(lg.Location, 'eastoutside') || strcmpi(lg.Location, 'bestoutside')
        lg.Location = 'best';
    end
end
end
