function [v_proc, v_proc_refine, t_proc, fs_proc, rms_thr] = preprocess_if_signal(data, cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 差频信号预处理：去直流、周期叠加平均、降采样、高通滤波
% 输入：data — load_measured_dataset 返回的结构体
%       cfg  — 配置结构体
% 输出：v_proc        — 高通滤波后的单周期信号
%       v_proc_refine — 附加低通滤波后的信号（边缘重建用，可为空）
%       t_proc        — 时间轴
%       fs_proc       — 处理后采样率
%       rms_thr       — RMS 门限（1% 峰值）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 去直流 + 周期叠加平均
v_raw = data.v_raw - mean(data.v_raw);
N_per   = round(cfg.T_m * data.fs);
N_sweep = floor(data.N_total / N_per);
v_mat   = reshape(v_raw(1:N_sweep * N_per), N_per, N_sweep);
v_avg   = mean(v_mat, 2);

fprintf('  每周期 %d 点, %d 个完整周期, 叠加平均 SNR+%.1f dB\n', ...
    N_per, N_sweep, 10*log10(N_sweep));

%% 降采样
ds = cfg.ds_factor;
v_ds    = v_avg(1:ds:end);
fs_proc = data.fs / ds;

%% 高通滤波
[b_hp, a_hp] = butter(2, cfg.f_hp_cut / (fs_proc / 2), 'high');
v_proc = filtfilt(b_hp, a_hp, v_ds);

%% 附加低通滤波（边缘重建用）
if cfg.f_refine_lp > 0
    [b_lp, a_lp] = butter(4, cfg.f_refine_lp / (fs_proc / 2), 'low');
    v_proc_refine = filtfilt(b_lp, a_lp, v_proc);
else
    v_proc_refine = v_proc;
end

%% 时间轴与门限
t_proc  = (0:length(v_proc)-1).' / fs_proc;
rms_thr = max(abs(v_proc)) * 0.01;

fprintf('  降采样 x%d -> fs_proc=%.0f MHz, N_proc=%d\n', ...
    ds, fs_proc/1e6, length(v_proc));

end
