% @[simulation/ADS/plot_all_results.m]
% Script to visualize ADS simulation results: Spectrum, Time Domain, and S21
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

files_sparam = {
    's21.txt',   'S21 幅值 (dB)';
    'delay.txt', '群延迟 (Delay)'
};

%% --- 3. Plot Spectrum (Figure 1) ---
figure('Name', 'ADS Spectrum Analysis', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

for i = 1:size(files_spectrum, 1)
    filename = files_spectrum{i, 1};
    title_str = files_spectrum{i, 2};
    
    [freq, dbm, ok] = load_ads_data(filename);
    
    if ok
        subplot(3, 1, i);
        
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
        title(title_str, 'FontSize', 12, 'FontName', font_cn, 'FontWeight', 'bold');
        grid on; axis tight;
    end
end

%% --- 4. Plot Time Domain (Figure 2) ---
figure('Name', 'ADS Time Domain Analysis', 'Color', 'w', 'Position', [150, 150, 1200, 800]);

for i = 1:size(files_time, 1)
    filename = files_time{i, 1};
    title_str = files_time{i, 2};
    
    [t, v, ok] = load_ads_data(filename);
    
    if ok
        subplot(3, 1, i);
        
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
        title(title_str, 'FontSize', 12, 'FontName', font_cn, 'FontWeight', 'bold');
        grid on; axis tight;
    end
end

%% --- 5. Plot S21 and Delay (Figure 3) ---
figure('Name', 'ADS S-Parameter Analysis', 'Color', 'w', 'Position', [200, 200, 1200, 800]);

for i = 1:size(files_sparam, 1)
    filename = files_sparam{i, 1};
    title_str = files_sparam{i, 2};
    [freq, data, ok] = load_ads_data(filename);
    
    if ok
        subplot(2, 1, i);
        
        if contains(filename, 'delay')
            ylabel('延迟 (ns)', 'FontSize', 11, 'FontName', font_cn);
            data = data * 1e9;
        else
            ylabel('幅值 (dB)', 'FontSize', 11, 'FontName', font_cn);
        end
        
        if max(freq) >= 1e9
            plot(freq/1e9, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
            xlabel('频率 (GHz)', 'FontSize', 11, 'FontName', font_cn);
        else
            plot(freq/1e6, data, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880]);
            xlabel('频率 (MHz)', 'FontSize', 11, 'FontName', font_cn);
        end
        
        title(title_str, 'FontSize', 13, 'FontName', font_cn, 'FontWeight', 'bold');
        grid on; axis tight;
    end
end

fprintf('所有绘图已完成。\n');

% --- 2. Helper Function to Load Data ---
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
