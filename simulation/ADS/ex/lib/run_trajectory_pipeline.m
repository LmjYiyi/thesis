function result = run_trajectory_pipeline(cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 时延轨迹提取通用管线
% 输入：cfg — 由 cfg_xxx.m 生成的配置结构体
% 输出：result — 包含各阶段结果的结构体
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_start = tic;

esp = esprit_extract();
pp  = trajectory_postprocess();

fprintf('===== %s =====\n', cfg.title_str);
fprintf('LFMCW: %.0f-%.0f GHz, B=%.1f GHz, T_m=%.0f us, K=%.2e Hz/s\n', ...
    cfg.f_start/1e9, cfg.f_end/1e9, cfg.B/1e9, cfg.T_m*1e6, cfg.K);

%% 1. 数据加载
data = load_measured_dataset(cfg);

%% 2. 预处理
[v_proc, v_proc_refine, t_proc, fs_proc, rms_thr] = preprocess_if_signal(data, cfg);

%% 3. 固定窗口粗提取
fprintf('\n【步骤1】固定窗口粗提取...\n');
base_raw = esp.run_fixed(v_proc, t_proc, fs_proc, cfg.f_start, cfg.K, ...
    rms_thr, cfg.cfg_base, cfg.f_valid_lo, cfg.f_beat_max, true);

base_clean = pp.clean(base_raw, true, cfg.cfg_base.name);
base_cal   = pp.calibrate(base_clean, cfg.K, true, cfg.cfg_base.name);
base_cal   = limit_frequency_range(base_cal, cfg.passband_lo, cfg.passband_hi);

result.base_raw   = base_raw;
result.base_clean = base_clean;
result.base_cal   = base_cal;

%% 4. 自适应窗口提取（可选）
if cfg.enable_adaptive
    fprintf('\n【步骤2】自适应分窗...\n');
    adapt_raw = esp.run_adaptive(v_proc, t_proc, fs_proc, cfg.f_start, cfg.K, ...
        rms_thr, base_clean, cfg.cfg_adapt, cfg.f_valid_lo, cfg.f_beat_max);

    adapt_clean = pp.clean(adapt_raw, true, cfg.cfg_adapt.name);

    result.adapt_raw   = adapt_raw;
    result.adapt_clean = adapt_clean;

    %% 5. 混合融合
    fprintf('\n【步骤3】混合融合...\n');
    hybrid_cal = pp.fuse(base_cal, adapt_clean, cfg.cfg_hybrid, true);
else
    hybrid_cal = base_cal;
    result.adapt_raw   = [];
    result.adapt_clean = [];
end

function out = limit_frequency_range(in, f_lo, f_hi)
out = in;

if ~isfield(in, 'f_probe') || isempty(in.f_probe)
    return;
end

mask = isfinite(in.f_probe) & in.f_probe >= f_lo & in.f_probe <= f_hi;
fields = fieldnames(in);
n = numel(in.f_probe);

for k = 1:numel(fields)
    name = fields{k};
    value = in.(name);

    if isnumeric(value) || islogical(value)
        if isvector(value) && numel(value) == n
            out.(name) = value(mask);
        elseif ndims(value) == 2 && size(value, 1) == n && size(value, 2) > 1
            out.(name) = value(mask, :);
        end
    elseif iscell(value) && isvector(value) && numel(value) == n
        out.(name) = value(mask);
    end
end
end

hybrid_cal = limit_frequency_range(hybrid_cal, cfg.passband_lo, cfg.passband_hi);

result.hybrid_cal = hybrid_cal;

%% 6. 边缘精细重建（可选）
if cfg.cfg_refine.enable
    switch cfg.cfg_refine.mode
        case 'data_driven'
            fprintf('\n【步骤4】右侧局部连续性重建...\n');
            % data_driven 模式使用低通滤波后的信号
            refined = refine_edge_segment(hybrid_cal, v_proc_refine, t_proc, fs_proc, ...
                cfg.f_start, cfg.K, rms_thr, base_cal, cfg.cfg_refine, ...
                cfg.f_valid_lo, cfg.f_beat_max, true);
        case 'mirror'
            fprintf('\n【步骤4】右侧边缘镜像重建...\n');
            refined = rebuild_right_edge(hybrid_cal, v_proc, t_proc, fs_proc, ...
                cfg.f_start, cfg.K, rms_thr, base_cal, cfg.cfg_refine, ...
                cfg.f_valid_lo, cfg.f_beat_max, true);
        case 'none'
            refined = hybrid_cal;
        otherwise
            warning('未知 refine mode: %s, 跳过重建', cfg.cfg_refine.mode);
            refined = hybrid_cal;
    end
    refined = limit_frequency_range(refined, cfg.passband_lo, cfg.passband_hi);
    result.refined = refined;
    result.final   = refined;
else
    result.refined = [];
    result.final   = hybrid_cal;
end

%% 7. 汇总输出
result.cfg  = cfg;
result.meta.elapsed_s = toc(t_start);
result.meta.n_base    = numel(base_cal.f_probe);
result.meta.n_final   = numel(result.final.f_probe);

fprintf('\n===== 结果汇总 =====\n');
fprintf('  固定窗口: %d 点', result.meta.n_base);
if cfg.enable_adaptive
    fprintf(', 自适应: %d 点', numel(result.adapt_clean.f_probe));
end
fprintf(', 最终: %d 点\n', result.meta.n_final);
fprintf('  频率: %.3f - %.3f GHz\n', ...
    min(result.final.f_probe)/1e9, max(result.final.f_probe)/1e9);
fprintf('  时延: %.2f - %.2f ns, 中位数 %.2f ns\n', ...
    min(result.final.tau)*1e9, max(result.final.tau)*1e9, ...
    median(result.final.tau)*1e9);
fprintf('  耗时: %.1f s\n', result.meta.elapsed_s);

end
