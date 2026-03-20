%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 实测时延轨迹提取（镜像版：固定窗 + 自适应窗 + 右边缘镜像对称重建）
% 右侧重建依赖左侧散点的镜像对称先验。
% 依赖文件：cfg_lowpass_filter.m, run_trajectory_pipeline.m,
%           plot_trajectory_result.m, rebuild_right_edge.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
addpath(fullfile(fileparts(mfilename('fullpath')), 'lib'));

%% 配置
cfg = cfg_lowpass_filter();

% 覆盖为镜像模式
cfg.cfg_refine.mode      = 'mirror';
cfg.cfg_refine.band_hi   = 37.55e9;   % 镜像版扫描范围略大
cfg.cfg_refine.tau_tol_lo = 0.35e-9;
cfg.cfg_refine.tau_tol_hi = 0.22e-9;
cfg.cfg_refine.purge_tol  = 0.40e-9;
cfg.cfg_refine.name       = '右侧边缘重建';

% 不需要 data_driven 专用字段，清除以避免歧义
if isfield(cfg.cfg_refine, 'group_freq_gap')
    cfg.cfg_refine = rmfield(cfg.cfg_refine, ...
        {'group_freq_gap', 'ref_span_lo', 'ref_span_hi', 'ref_min_points', ...
         'consensus_min', 'edge_uplift_gain', 'edge_uplift_power', 'edge_uplift_cap'});
end

cfg.export_name = 'exp_delay_trajectory_mirror';
cfg.title_str   = '实测 LFMCW 时延轨迹（镜像版）';

%% 运行管线
result = run_trajectory_pipeline(cfg);

%% 绘图与导出
plot_trajectory_result(result, cfg);
