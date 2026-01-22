%% 图3-4: 电子密度敏感性分析 - CST全波仿真 (固定碰撞频率, 改变截止频率)
% 数据来源: CST仿真 (guding_nue.txt)
% 对应章节: 3.2.2 截止频率附近的群时延渐近发散特征
% 曲线标签来自原始文件: 27.5E9, 29E9, 30.4E9 (截止频率 GHz)
% 与图3-3(a)理论计算对比,展示多径振荡效应

clc; clear; close all;

%% 1. 读取CST仿真数据
cst_file = '../../simulation/cst_data/guding_nue.txt';
[f_data, tau_data, labels] = read_cst_multicolumn(cst_file);

%% 2. 绘图
figure('Name', '图3-4: 电子密度敏感性(CST)', 'Color', 'w', 'Position', [100 100 800 550]);
hold on; grid on;

% 定义颜色和线型 (按截止频率从小到大排列)
colors = {[0 0.4 0.8], [0.8 0 0], [0 0.6 0]};  % 蓝, 红, 绿
line_styles = {'-', '-', '-'};
line_widths = [1.8, 1.8, 1.8];

% 提取截止频率值并排序
fp_values = [];
for i = 1:length(labels)
    % 从标签提取截止频率数值 (如 "27.5E9" -> 27.5)
    label_str = labels{i};
    fp_val = str2double(regexprep(label_str, 'E9.*', ''));
    if ~isnan(fp_val)
        fp_values(end+1) = fp_val;
    end
end

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
    
    % 构造图例字符串 (从标签中提取截止频率)
    fp_str = regexprep(label_str, 'E9.*', '');
    legend_str{end+1} = ['f_p = ' fp_str ' GHz'];
end

xlabel('探测频率 f (GHz)', 'FontSize', 12);
ylabel('群时延 \tau_g (ns)', 'FontSize', 12);
title('图3-4 固定碰撞频率条件下不同电子密度的群时延曲线 (CST全波仿真)', 'FontSize', 13);

xlim([20 40]);
ylim([-2 12]);

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
