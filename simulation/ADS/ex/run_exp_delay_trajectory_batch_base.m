%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 批量运行三组实测数据的原始固定窗口时延轨迹提取
% 用途：
% 1. 依次运行 lowpassfilter_filter*.csv 三份数据
% 2. 调用原始 ex/extract_exp_delay_trajectory.m
% 3. 在终端输出各数据集的散点数量、时延范围和频率范围
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
summary_dir = fullfile(script_dir, '..', 'figures_export');
if ~exist(summary_dir, 'dir')
    mkdir(summary_dir);
end
summary_file = fullfile(summary_dir, 'exp_delay_trajectory_base_summary.md');

%% 1. 批处理配置
data_file_list = { ...
    'lowpassfilter_filter.csv', ...
    'lowpassfilter_filter_1.csv', ...
    'lowpassfilter_filter_2.csv' ...
    };

case_title_list = { ...
    '原始数据', ...
    '实测数据 1', ...
    '实测数据 2' ...
    };

out_suffix_list = { ...
    '_base', ...
    '_1', ...
    '_2' ...
    };

batch_summaries = cell(numel(data_file_list), 1);

%% 2. 逐份运行原始脚本
keep_console = true;
for i_case = 1:numel(data_file_list)
    fprintf('\n============================================================\n');
    fprintf('【批处理】%d/%d : %s\n', ...
        i_case, numel(data_file_list), case_title_list{i_case});
    fprintf('  数据文件: %s\n', data_file_list{i_case});
    fprintf('============================================================\n');

    data_file = data_file_list{i_case};
    case_title = case_title_list{i_case};
    out_suffix = out_suffix_list{i_case};

    run(fullfile(script_dir, 'extract_exp_delay_trajectory.m'));
    batch_summaries{i_case} = result_summary;
end

%% 3. 汇总输出
fprintf('\n================ 原始固定窗口批处理汇总 ================\n');
fprintf('  %-10s %-10s %-12s %-12s %-12s %-12s\n', ...
    '数据标签', '点数', 'tau_min(ns)', 'tau_max(ns)', 'f_min(GHz)', 'f_max(GHz)');

for i_case = 1:numel(batch_summaries)
    summary_i = batch_summaries{i_case};
    fprintf('  %-10s %-10d %-12.2f %-12.2f %-12.3f %-12.3f\n', ...
        summary_i.case_title, ...
        summary_i.point_count, ...
        summary_i.tau_min_ns, ...
        summary_i.tau_max_ns, ...
        summary_i.f_min_ghz, ...
        summary_i.f_max_ghz);
end

fprintf('\n导出文件命名后缀对应关系：\n');
for i_case = 1:numel(out_suffix_list)
    fprintf('  %-10s -> exp_delay_trajectory%s.tiff\n', ...
        case_title_list{i_case}, out_suffix_list{i_case});
end

%% 4. 写入汇总文件
fid = fopen(summary_file, 'w', 'n', 'UTF-8');
fprintf(fid, '# 原始固定窗口三组实测数据汇总\n\n');
fprintf(fid, '| 数据标签 | 点数 | tau_min(ns) | tau_max(ns) | f_min(GHz) | f_max(GHz) |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');

for i_case = 1:numel(batch_summaries)
    summary_i = batch_summaries{i_case};
    fprintf(fid, '| %s | %d | %.2f | %.2f | %.3f | %.3f |\n', ...
        summary_i.case_title, ...
        summary_i.point_count, ...
        summary_i.tau_min_ns, ...
        summary_i.tau_max_ns, ...
        summary_i.f_min_ghz, ...
        summary_i.f_max_ghz);
end

fprintf(fid, '\n## 图像文件\n\n');
for i_case = 1:numel(out_suffix_list)
    fprintf(fid, '- %s: `exp_delay_trajectory%s.tiff`\n', ...
        case_title_list{i_case}, out_suffix_list{i_case});
end

fclose(fid);
fprintf('\n汇总文件已写入: %s\n', summary_file);
