function data = load_measured_dataset(cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 加载实测数据集，返回统一结构体
% 输入：cfg — 由 cfg_xxx.m 生成的配置结构体
% 输出：data.t_raw, data.v_raw, data.fs, data.N_total
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lib_dir    = fileparts(mfilename('fullpath'));
script_dir = fileparts(lib_dir);              % ex/ 根目录
data_file  = fullfile(script_dir, 'data', cfg.data_file);

fprintf('\n正在加载数据: %s\n', cfg.data_file);
raw = readmatrix(data_file);
data.t_raw   = raw(:, 1);
data.v_raw   = raw(:, 2);

dt = median(diff(data.t_raw));
data.fs      = round(1 / dt);
data.N_total = length(data.t_raw);

fprintf('  采样率: %.0f MHz, 总点数: %d, 时长: %.2f ms\n', ...
    data.fs / 1e6, data.N_total, (data.t_raw(end) - data.t_raw(1)) * 1e3);

end
