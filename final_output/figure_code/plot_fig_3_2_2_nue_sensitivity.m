%% 图3-5: 碰撞频率敏感性分析 - CST全波仿真 (固定截止频率, 改变碰撞频率)
% 数据来源: CST仿真 (guding_wp.txt)
% 对应章节: 3.2.2 碰撞频率对群时延的二阶微扰效应
% 曲线标签来自原始文件: 28.98E9_1.5E9, 28.98E9_3E9, 28.98E9_5E9 
% 格式: 截止频率_碰撞频率 (单位: Hz)
% 与图3-3(b)理论计算对比,展示多径振荡效应

clc; clear; close all;

%% 1. 读取CST仿真数据
cst_file = '../../simulation/cst_data/guding_wp.txt';
[f_data, tau_data, labels] = read_cst_multicolumn(cst_file);

%% 2. 绘图
figure('Name', '碰撞频率敏感性(CST)', 'Color', 'w', 'Position', [100 100 800 550]);
hold on; grid on;

% 定义颜色和线型
colors = {[0 0.5 0], [0 0 0.8], [0.8 0 0.5]};  % 绿, 蓝, 品红
line_styles = {'-', '-', '-'};
line_widths = [1.8, 1.8, 1.8];

% 绘制所有曲线
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
        'LineWidth', line_widths(color_idx), ...
        'LineStyle', line_styles{color_idx});
    h_plots = [h_plots, h];
    
    % 构造图例字符串 (从标签中提取碰撞频率)
    % 标签格式: "28.98E9_1.5E9" -> 提取 "_" 后面的数值
    parts = strsplit(label_str, '_');
    if length(parts) >= 2
        nu_str = regexprep(parts{2}, 'E9.*', '');
        legend_str{end+1} = ['\nu_e = ' nu_str ' GHz'];
    else
        legend_str{end+1} = label_str;
    end
end

xlabel('探测频率 f (GHz)', 'FontSize', 12);
ylabel('群时延 \tau_g (ns)', 'FontSize', 12);
title('固定等离子体频率条件下不同碰撞频率的群时延曲线 (CST全波仿真)', 'FontSize', 13);

xlim([20 40]);
ylim([0 12]);

legend(h_plots, legend_str, 'Location', 'NorthEast', 'FontSize', 10);
set(gca, 'FontSize', 11);

%% 辅助函数: 读取CST多列群时延数据
function [freq, data, labels] = read_cst_multicolumn(filename)
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
    
    % 使用第一列的频率作为参考
    freq = all_freq{1};
    data = zeros(length(freq), length(all_data));
    
    for i = 1:length(all_data)
        % 插值对齐到参考频率轴
        if length(all_freq{i}) > 1
            data(:,i) = interp1(all_freq{i}, all_data{i}, freq, 'linear', NaN);
        end
    end
end
