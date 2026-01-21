function generate_lfmcw_excitation(filename, f_start, f_end, T_m, f_s, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 生成LFMCW激励信号并保存为CST可读格式
%
% 输入:
%   filename - 输出文件路径
%   f_start  - 起始频率 (Hz)
%   f_end    - 终止频率 (Hz)
%   T_m      - 扫频周期 (s)
%   f_s      - 采样率 (Hz)
%
% 可选参数:
%   'Format'   - 'CST' (默认) 或 'ASCII'
%   'TimeUnit' - 'ns', 'us', 's' (默认 'ns')
%   'Amplitude' - 信号幅度 (默认 1.0)
%
% 作者: Auto-generated for thesis project
% 日期: 2026-01-19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 解析可选参数
p = inputParser;
addParameter(p, 'Format', 'CST');
addParameter(p, 'TimeUnit', 'ns');
addParameter(p, 'Amplitude', 1.0);
parse(p, varargin{:});

format_type = p.Results.Format;
time_unit = p.Results.TimeUnit;
amplitude = p.Results.Amplitude;

%% 生成时间轴
t = 0 : 1/f_s : T_m;
N = length(t);

%% 计算LFMCW信号
K = (f_end - f_start) / T_m;  % 调频斜率

% 瞬时相位
phi = 2*pi*f_start*t + pi*K*t.^2;

% LFMCW信号
s = amplitude * cos(phi);

%% 时间单位转换
switch time_unit
    case 'ns'
        t_output = t * 1e9;
        time_unit_str = 'ns';
    case 'us'
        t_output = t * 1e6;
        time_unit_str = 'us';
    case 's'
        t_output = t;
        time_unit_str = 's';
    otherwise
        t_output = t * 1e9;
        time_unit_str = 'ns';
end

%% 保存文件
fid = fopen(filename, 'w');
if fid == -1
    error('无法创建文件: %s', filename);
end

% 写入头部注释
fprintf(fid, '%% LFMCW Excitation Signal for CST Time Domain Solver\n');
fprintf(fid, '%% Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '%% \n');
fprintf(fid, '%% Parameters:\n');
fprintf(fid, '%%   f_start = %.3f GHz\n', f_start/1e9);
fprintf(fid, '%%   f_end   = %.3f GHz\n', f_end/1e9);
fprintf(fid, '%%   T_m     = %.3f us\n', T_m*1e6);
fprintf(fid, '%%   f_s     = %.3f GHz\n', f_s/1e9);
fprintf(fid, '%%   K       = %.3f MHz/us\n', K/1e12);
fprintf(fid, '%%   N       = %d samples\n', N);
fprintf(fid, '%% \n');
fprintf(fid, '%% Format: time (%s)  amplitude\n', time_unit_str);
fprintf(fid, '%% \n');

% 写入数据
for i = 1:N
    fprintf(fid, '%.9e  %.9e\n', t_output(i), s(i));
end

fclose(fid);

fprintf('✓ LFMCW激励信号已生成:\n');
fprintf('  文件: %s\n', filename);
fprintf('  采样点数: %d\n', N);
fprintf('  时长: %.3f %s\n', t_output(end), time_unit_str);
fprintf('  频率范围: %.2f - %.2f GHz\n', f_start/1e9, f_end/1e9);

end


%% 测试用例 (取消注释运行)
% if false
%     % 示例调用
%     generate_lfmcw_excitation('test_lfmcw.sig', ...
%         34.2e9, 37.4e9, ...  % 34.2 - 37.4 GHz
%         50e-6, ...           % 50 us
%         80e9, ...            % 80 GHz 采样率
%         'TimeUnit', 'ns');
% end
