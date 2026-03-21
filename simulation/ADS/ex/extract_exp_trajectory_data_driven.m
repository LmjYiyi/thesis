%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时延轨迹提取（纯数据驱动版）
% 右侧边缘重建完全基于局部连续性与多窗口共识，不使用任何镜像先验。
% 依赖文件：cfg_lowpass_filter.m, run_trajectory_pipeline.m,
%           plot_trajectory_result.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), 'lib'));

%% 配置
cfg = cfg_lowpass_filter();

% 覆盖默认参数以匹配 data_driven 模式
cfg.cfg_refine.mode  = 'data_driven';
cfg.cfg_refine.name  = '右侧局部连续性重建';

cfg.export_name = 'exp_delay_trajectory_data_driven';
cfg.xlim_range  = [36.4, 37.6];
cfg.deviation_report.enable = true;
cfg.deviation_report.top_n  = 8;
cfg.title_str   = '实测 LFMCW 时延轨迹（纯数据驱动版）';

%% 运行管线
result = run_trajectory_pipeline(cfg);

%% 绘图与导出
plot_trajectory_result(result, cfg);
