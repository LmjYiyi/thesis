function cfg = cfg_lowpass_filter()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 数据集配置：HXLBQ-DTA1329 低通滤波器（ADS 实测）
% 返回完整的管线参数结构体，所有频率边界由通带参数自动推导。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LFMCW 参数
cfg.f_start = 34e9;
cfg.f_end   = 37e9;
cfg.B       = cfg.f_end - cfg.f_start;
cfg.T_m     = 50e-6;
cfg.K       = cfg.B / cfg.T_m;

%% 数据集
cfg.dataset_name = 'HXLBQ-DTA1329 lowpass filter';
cfg.data_file    = 'lowpassfilter_filter.csv';   % 相对于 data/ 子目录

%% 通带参数（所有区域边界由此推导）
cfg.passband_lo  = 36.5e9;
cfg.passband_hi  = 37.5e9;
cfg.center_freq  = 37.0e9;
BW = cfg.passband_hi - cfg.passband_lo;

%% 预处理
cfg.ds_factor    = 10;
cfg.f_hp_cut     = 10e3;
cfg.f_refine_lp  = 0.80e6;     % 右侧重建用的低通截止，0 表示不启用

%% ESPRIT 信号门限
cfg.f_valid_lo   = 20e3;
cfg.f_beat_max   = 300e3;

%% 固定窗口配置
cfg.cfg_base.win_len  = 150;
cfg.cfg_base.step_len = 13;
cfg.cfg_base.L_sub    = 75;
cfg.cfg_base.name     = '固定窗口';

%% 自适应窗口配置
cfg.cfg_adapt.step_center = 9;
cfg.cfg_adapt.win_short   = 140;
cfg.cfg_adapt.win_mid1    = 210;
cfg.cfg_adapt.win_mid2    = 260;
cfg.cfg_adapt.win_long    = 300;
cfg.cfg_adapt.tau_thr_1   = 0.9e-9;
cfg.cfg_adapt.tau_thr_2   = 1.70e-9;
cfg.cfg_adapt.tau_thr_3   = 2.25e-9;
cfg.cfg_adapt.name        = '自适应窗口';

%% 混合融合配置（由通带 + 相对偏移推导）
cfg.cfg_hybrid.f_flat_lo    = cfg.passband_lo + 0.28 * BW;   % 36.78e9
cfg.cfg_hybrid.f_flat_hi    = cfg.passband_hi - 0.12 * BW;   % 37.38e9
cfg.cfg_hybrid.mid_fill_gap = 0.020e9;
cfg.cfg_hybrid.local_cap.enable = true;
cfg.cfg_hybrid.local_cap.margin_s = [
    0.162e-9;
    0.290e-9;
    0.076e-9
];
cfg.cfg_hybrid.local_cap.ref_pad_hz = 0.14e9;
cfg.cfg_hybrid.local_cap.bands_hz = [
    36.79e9, 36.84e9;
    36.84e9, 36.9527e9;
    37.20e9, 37.30e9
];

%% 右侧边缘重建配置（默认：data_driven 模式）
cfg.cfg_refine.enable        = true;
cfg.cfg_refine.mode          = 'data_driven';   % 'none' | 'data_driven' | 'mirror'
cfg.cfg_refine.band_lo       = cfg.passband_hi - 0.12 * BW;   % 37.38e9
cfg.cfg_refine.band_hi       = cfg.passband_hi;                % 37.50e9
cfg.cfg_refine.win_lens      = [130, 100, 80];
cfg.cfg_refine.L_sub_ratios  = [1/2, 1/3, 2/5];
cfg.cfg_refine.step_len      = 3;
cfg.cfg_refine.min_freq_gap  = 0.003e9;

% data_driven 专用参数
cfg.cfg_refine.group_freq_gap   = 0.004e9;
cfg.cfg_refine.ref_span_lo      = 0.24e9;
cfg.cfg_refine.ref_span_hi      = 0.00e9;
cfg.cfg_refine.ref_min_points   = 5;
cfg.cfg_refine.consensus_min    = 2;
cfg.cfg_refine.edge_uplift_gain  = 0.95;
cfg.cfg_refine.edge_uplift_power = 0.85;
cfg.cfg_refine.edge_uplift_cap   = 0.28e-9;
cfg.cfg_refine.tau_tol_lo       = 0.32e-9;
cfg.cfg_refine.tau_tol_hi       = 0.36e-9;
cfg.cfg_refine.purge_band_lo    = cfg.passband_hi - 0.20 * BW;  % 37.30e9
cfg.cfg_refine.purge_tol        = 0.36e-9;
cfg.cfg_refine.name             = '右侧局部连续性重建';

%% 启用自适应提取（baseline 模式关闭此开关）
cfg.enable_adaptive = true;

%% 频率区域定义（由通带自动推导）
cfg.regions.left_edge       = [cfg.passband_lo,                cfg.passband_lo + 0.12 * BW];
cfg.regions.left_shoulder   = [cfg.passband_lo + 0.12 * BW,   cfg.passband_lo + 0.28 * BW];
cfg.regions.flat_mid        = [cfg.center_freq - 0.22 * BW,   cfg.center_freq + 0.22 * BW];
cfg.regions.right_shoulder  = [cfg.passband_hi - 0.28 * BW,   cfg.passband_hi - 0.12 * BW];
cfg.regions.right_edge      = [cfg.passband_hi - 0.12 * BW,   cfg.passband_hi];

%% 参考曲线配置
cfg.reference.type = 'none';   % 'none' | 's2p' | 'manual'

%% 绘图 / 导出
cfg.export_name = 'exp_delay_trajectory';
cfg.title_str   = '实测 LFMCW 时延轨迹';
cfg.xlim_range  = [36.45, 37.55];

end
