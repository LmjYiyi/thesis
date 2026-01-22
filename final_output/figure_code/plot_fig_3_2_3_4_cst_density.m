%% 图3-6 & 图3-7: CST仿真的低/高电子密度群时延曲线
% 数据来源: low_density.txt, high_density.txt
% 对应章节: 3.2.2 群时延曲线随电子密度的演化规律总结
% 
% low_density.txt 曲线标签: 1e17, 1e18, 5e18 (电子密度 m^-3)
% high_density.txt 曲线标签: 7e18, 1.2e19, 1.4e19 (电子密度 m^-3)

clc; clear; close all;

%% ======================== 图3-6: 低电子密度 ========================
figure('Name', '图3-6: 低电子密度CST仿真', 'Color', 'w', 'Position', [50 100 750 500]);

% 读取数据
cst_file = '../../simulation/cst_data/low_density.txt';
[f_data, tau_data, labels] = read_cst_multicolumn(cst_file);

hold on; grid on;

% 定义颜色 (对应不同密度, 颜色从浅到深表示密度增加)
colors = {[0.2 0.6 1.0], [0 0.4 0.8], [0 0.2 0.5]};  % 浅蓝 -> 深蓝
line_widths = [1.8, 1.8, 1.8];

% 绘制所有有效曲线
valid_curves = 0;
legend_str = {};
h_plots = [];

for i = 1:size(tau_data, 2)
    label_str = labels{i};
    
    % 跳过 null 列
    if contains(lower(label_str), 'null')
        continue;
    end
    
    valid_curves = valid_curves + 1;
    color_idx = mod(valid_curves - 1, length(colors)) + 1;
    
    h = plot(f_data, tau_data(:,i), ...
        'Color', colors{color_idx}, ...
        'LineWidth', line_widths(color_idx));
    h_plots = [h_plots, h];
    
    % 构造图例字符串
    legend_str{end+1} = ['n_e = ' label_str ' m^{-3}'];
end

xlabel('探测频率 f (GHz)', 'FontSize', 12);
ylabel('群时延 \tau_g (ns)', 'FontSize', 12);
title('图3-6 低电子密度区间CST仿真群时延曲线', 'FontSize', 14);

xlim([26 40]);
ylim([4 8]);

legend(h_plots, legend_str, 'Location', 'NorthEast', 'FontSize', 10);
set(gca, 'FontSize', 11);

%% ======================== 图3-7: 高电子密度 ========================
figure('Name', '图3-7: 高电子密度CST仿真', 'Color', 'w', 'Position', [850 100 750 500]);

% 读取数据
cst_file = '../../simulation/cst_data/high_density.txt';
[f_data, tau_data, labels] = read_cst_multicolumn(cst_file);

hold on; grid on;

% 定义颜色 (对应不同密度)
colors = {[0.2 0.7 0.3], [0.8 0.4 0], [0.8 0 0]};  % 绿, 橙, 红

% 绘制所有有效曲线
valid_curves = 0;
legend_str = {};
h_plots = [];

for i = 1:size(tau_data, 2)
    label_str = labels{i};
    
    % 跳过 null 列
    if contains(lower(label_str), 'null')
        continue;
    end
    
    valid_curves = valid_curves + 1;
    color_idx = mod(valid_curves - 1, length(colors)) + 1;
    
    h = plot(f_data, tau_data(:,i), ...
        'Color', colors{color_idx}, ...
        'LineWidth', 1.8);
    h_plots = [h_plots, h];
    
    % 构造图例字符串
    legend_str{end+1} = ['n_e = ' label_str ' m^{-3}'];
end

xlabel('探测频率 f (GHz)', 'FontSize', 12);
ylabel('群时延 \tau_g (ns)', 'FontSize', 12);
title('图3-7 高电子密度区间CST仿真群时延曲线', 'FontSize', 14);

xlim([26 40]);
ylim([-5 12]);

legend(h_plots, legend_str, 'Location', 'NorthEast', 'FontSize', 10);
set(gca, 'FontSize', 11);

%% ======================== 辅助函数 ========================
function [freq, data, labels] = read_cst_multicolumn(filename)
    % 读取CST导出的多数据块群时延文件
    % 每个数据块以 #"Frequency / GHz" 开头
    
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    % 存储各列数据
    all_freq = {};
    all_data = {};
    labels = {};
    
    current_freq = [];
    current_data = [];
    current_label = '';
    
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line)
            continue;
        end
        
        % 检测新数据块的开始 (标签行)
        if startsWith(line, '#"Frequency')
            % 保存之前的数据块
            if ~isempty(current_freq)
                all_freq{end+1} = current_freq;
                all_data{end+1} = current_data;
                labels{end+1} = current_label;
            end
            
            % 解析新标签
            tokens = strsplit(line, '\t');
            if length(tokens) >= 2
                label = strtrim(tokens{2});
                label = strrep(label, '"', '');
                label = strrep(label, ' [Real Part]', '');
                current_label = label;
            end
            
            current_freq = [];
            current_data = [];
            continue;
        end
        
        % 跳过参数行和分隔线
        if startsWith(line, '#')
            continue;
        end
        
        % 读取数据行
        values = sscanf(line, '%f %f');
        if length(values) == 2
            current_freq = [current_freq; values(1)];
            current_data = [current_data; values(2)];
        end
    end
    
    % 保存最后一个数据块
    if ~isempty(current_freq)
        all_freq{end+1} = current_freq;
        all_data{end+1} = current_data;
        labels{end+1} = current_label;
    end
    
    fclose(fid);
    
    % 将所有数据对齐到共同频率轴
    if isempty(all_freq)
        freq = [];
        data = [];
        return;
    end
    
    % 找出所有频率的最大公共范围
    f_min = -inf;
    f_max = inf;
    for i = 1:length(all_freq)
        f_min = max(f_min, min(all_freq{i}));
        f_max = min(f_max, max(all_freq{i}));
    end
    
    % 创建统一频率轴
    freq = linspace(f_min, f_max, 1000)';
    data = zeros(length(freq), length(all_data));
    
    for i = 1:length(all_data)
        % 插值对齐到参考频率轴
        if length(all_freq{i}) > 1
            data(:,i) = interp1(all_freq{i}, all_data{i}, freq, 'linear', NaN);
        end
    end
end
