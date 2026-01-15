%% save_all_figures.m
% 论文图片统一保存脚本
% 功能：运行所有绑图脚本并将图片保存到指定目录
% 使用方法：在MATLAB中运行此脚本
% 
% 注意事项：
% 1. 请确保当前工作目录为 final_output/figure_code/
% 2. 图片将保存至 final_output/figures/ 目录

clear; clc; close all;

%% 配置
% 输出目录（相对路径）
output_dir = '../figures';

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('已创建输出目录: %s\n', output_dir);
end

% 图片列表（脚本名 -> 输出文件名）
figure_scripts = {
    'plot_fig_3_3a', '图3-3a_电子密度对群时延曲线的拓扑控制';
    'plot_fig_3_3b', '图3-3b_碰撞频率解耦特性';
    'plot_fig_3_4',  '图3-4_时延演化轨迹对比';
    'plot_fig_3_5',  '图3-5_频谱散焦效应';
    'plot_fig_3_6',  '图3-6_带宽散焦耦合曲线';
    'plot_fig_3_7',  '图3-7_允许带宽与截止频率参数空间';
    'plot_fig_4_3',  '图4-3_修正项影响对比';
    'plot_fig_4_4',  '图4-4_加权矩阵频率分布';
    'plot_fig_4_5',  '图4-5_LM收敛轨迹';
    'plot_fig_4_6',  '图4-6_特征提取方法对比';
    'plot_fig_4_7',  '图4-7_不同电子密度拟合结果';
    'plot_fig_4_8',  '图4-8_拟合残差频率分布';
    'plot_fig_4_9',  '图4-9_碰撞频率失配鲁棒性';
    'plot_fig_4_10', '图4-10_不同碰撞频率先验拟合对比';
};

%% 执行绑图并保存
fprintf('\n========== 开始批量生成论文图片 ==========\n\n');

success_count = 0;
fail_count = 0;
failed_scripts = {};

for i = 1:size(figure_scripts, 1)
    script_name = figure_scripts{i, 1};
    output_name = figure_scripts{i, 2};
    
    fprintf('[%2d/%2d] 正在处理: %s\n', i, size(figure_scripts, 1), script_name);
    
    try
        % 运行绑图脚本
        run(script_name);
        
        % 获取当前图形句柄
        fig = gcf;
        
        % 保存为 PNG（高分辨率，用于Word/预览）
        png_path = fullfile(output_dir, [output_name, '.png']);
        print(fig, '-dpng', '-r300', png_path);
        
        % 保存为 SVG（矢量图，用于LaTeX排版）
        svg_path = fullfile(output_dir, [output_name, '.svg']);
        print(fig, '-dsvg', svg_path);
        
        fprintf('       ✓ 已保存: %s.png/.svg\n', output_name);
        success_count = success_count + 1;
        
        % 关闭当前图形
        close(fig);
        
    catch ME
        fprintf('       ✗ 失败: %s\n', ME.message);
        fail_count = fail_count + 1;
        failed_scripts{end+1} = script_name; %#ok<SAGROW>
    end
    
    fprintf('\n');
end

%% 输出统计
fprintf('========== 批量生成完成 ==========\n\n');
fprintf('成功: %d 个\n', success_count);
fprintf('失败: %d 个\n', fail_count);

if ~isempty(failed_scripts)
    fprintf('\n失败的脚本:\n');
    for i = 1:length(failed_scripts)
        fprintf('  - %s\n', failed_scripts{i});
    end
end

fprintf('\n图片保存目录: %s\n', fullfile(pwd, output_dir));
fprintf('========================================\n');
