%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时延轨迹提取（基线版：固定窗口粗提取 + 双锚点校准）
% 无右侧边缘重建，无自适应分窗，仅做滑动窗口 MDL-ESPRIT + 后处理。
% 依赖文件：cfg_lowpass_filter.m, run_trajectory_pipeline.m,
%           plot_trajectory_result.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), 'lib'));

%% 配置
cfg = cfg_lowpass_filter();

% 关闭自适应与重建
cfg.enable_adaptive     = false;
cfg.cfg_refine.enable   = false;
cfg.cfg_refine.mode     = 'none';

cfg.export_name = 'exp_delay_trajectory_baseline';
cfg.title_str   = '实测 LFMCW 时延轨迹（基线版）';
cfg.xlim_range  = [36.0, 38.0];

%% 运行管线
result = run_trajectory_pipeline(cfg);

%% 绘图与导出
plot_trajectory_result(result, cfg);
