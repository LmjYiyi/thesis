%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST 时域仿真全自动脚本
% 功能: MATLAB全自动控制CST进行建模、时域仿真、信号导出和反演
%
% 流程:
%   1. 生成LFMCW激励信号
%   2. 连接CST并创建CSRR模型
%   3. 配置时域求解器 + 用户自定义激励
%   4. 运行仿真
%   5. 导出端口信号
%   6. MATLAB混频处理和参数反演
%
% 作者: Auto-generated for thesis project
% 日期: 2026-01-19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 0. 初始化
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
data_dir = fullfile(script_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

fprintf('================================================\n');
fprintf('CST 时域仿真全自动脚本\n');
fprintf('================================================\n\n');

%% 1. 参数配置
% -----------------------
% 1.1 LFMCW信号参数
% -----------------------
params.lfmcw.f_start = 34.2e9;    % 起始频率 (Hz)
params.lfmcw.f_end = 37.4e9;      % 终止频率 (Hz)
params.lfmcw.T_m = 5e-6;          % 扫频周期 (s) - 缩短以节省时间
params.lfmcw.f_s = 100e9;         % 采样率 (Hz)

% 派生参数
params.lfmcw.B = params.lfmcw.f_end - params.lfmcw.f_start;
params.lfmcw.K = params.lfmcw.B / params.lfmcw.T_m;

% -----------------------
% 1.2 CST模型参数
% -----------------------
params.cst.version = '2023';      % CST版本
params.cst.project_name = 'CSRR_TimeDomain';

% WR-28 波导
params.wg.a = 7.112;              % 宽度 (mm)
params.wg.b = 3.556;              % 高度 (mm)
params.wg.length = 20;            % 长度 (mm)

% CSRR参数
params.csrr.L_out = 2.0;          % 外环边长 (mm)
params.csrr.L_in = 1.4;           % 内环边长 (mm)
params.csrr.z_pos = 10;           % 位置 (mm)

% 仿真参数
params.sim.f_min = 30;            % GHz
params.sim.f_max = 40;            % GHz

% 物理常数
c = 3e8;

fprintf('【参数配置】\n');
fprintf('LFMCW: %.2f - %.2f GHz, T_m = %.1f μs\n', ...
    params.lfmcw.f_start/1e9, params.lfmcw.f_end/1e9, params.lfmcw.T_m*1e6);
fprintf('调频斜率 K = %.2f GHz/μs\n', params.lfmcw.K/1e15);
fprintf('\n');

%% 2. 生成LFMCW激励信号
fprintf('【步骤1/6】生成LFMCW激励信号...\n');

t = 0 : 1/params.lfmcw.f_s : params.lfmcw.T_m;
N = length(t);
phi = 2*pi*params.lfmcw.f_start*t + pi*params.lfmcw.K*t.^2;
s_tx = cos(phi);

% 保存为CST格式
excitation_file = fullfile(data_dir, 'lfmcw_excitation.sig');
fid = fopen(excitation_file, 'w');
for i = 1:N
    fprintf(fid, '%.9e  %.9e\n', t(i)*1e9, s_tx(i));  % 时间单位ns
end
fclose(fid);

fprintf('  ✓ 激励信号已保存: %s\n', excitation_file);
fprintf('  采样点数: %d, 时长: %.2f μs\n', N, params.lfmcw.T_m*1e6);

%% 3. 连接CST并创建模型
fprintf('\n【步骤2/6】连接CST Microwave Studio...\n');

try
    cst = actxserver('CSTStudio.Application');
    fprintf('  ✓ CST连接成功!\n');
    
    mws = cst.invoke('NewMWS');
    fprintf('  ✓ 新项目创建成功!\n');
    
    CST_CONNECTED = true;
catch ME
    fprintf('  ✗ CST连接失败: %s\n', ME.message);
    fprintf('  将使用模拟数据继续演示...\n');
    CST_CONNECTED = false;
end

%% 4. 创建CSRR模型 + 配置时域求解器
if CST_CONNECTED
    fprintf('\n【步骤3/6】创建CSRR模型并配置时域求解器...\n');
    
    try
        % 4.1 设置单位
        vba_units = [...
            'With Units' newline ...
            '  .Geometry "mm"' newline ...
            '  .Frequency "GHz"' newline ...
            '  .Time "ns"' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'define units', vba_units);
        fprintf('  ✓ 单位设置完成\n');
        
        % 4.2 设置频率范围
        vba_freq = sprintf('Solver.FrequencyRange "%.2f", "%.2f"', ...
            params.sim.f_min, params.sim.f_max);
        invoke(mws, 'AddToHistory', 'define frequency', vba_freq);
        fprintf('  ✓ 频率范围设置完成\n');
        
        % 4.3 创建材料
        vba_mat = [...
            'With Material' newline ...
            '  .Reset' newline ...
            '  .Name "Rogers5880"' newline ...
            '  .Type "Normal"' newline ...
            '  .Epsilon "2.2"' newline ...
            '  .TanD "0.0009"' newline ...
            '  .Colour "0.8", "0.6", "0.2"' newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'define material', vba_mat);
        fprintf('  ✓ 材料定义完成\n');
        
        % 4.4 创建波导
        vba_wg = [...
            'With Brick' newline ...
            '  .Reset' newline ...
            '  .Name "Air_Cavity"' newline ...
            '  .Component "Waveguide"' newline ...
            '  .Material "Vacuum"' newline ...
            sprintf('  .Xrange "%.4f", "%.4f"', -params.wg.a/2, params.wg.a/2) newline ...
            sprintf('  .Yrange "%.4f", "%.4f"', -params.wg.b/2, params.wg.b/2) newline ...
            sprintf('  .Zrange "0", "%.4f"', params.wg.length) newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'create waveguide', vba_wg);
        fprintf('  ✓ 波导创建完成\n');
        
        % 4.5 创建CSRR外环
        y_bottom = -params.wg.b/2;
        vba_csrr_out = [...
            'With Brick' newline ...
            '  .Reset' newline ...
            '  .Name "CSRR_Outer"' newline ...
            '  .Component "CSRR"' newline ...
            '  .Material "PEC"' newline ...
            sprintf('  .Xrange "%.4f", "%.4f"', -params.csrr.L_out/2, params.csrr.L_out/2) newline ...
            sprintf('  .Yrange "%.4f", "%.4f"', y_bottom, y_bottom + 0.035) newline ...
            sprintf('  .Zrange "%.4f", "%.4f"', params.csrr.z_pos - params.csrr.L_out/2, params.csrr.z_pos + params.csrr.L_out/2) newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'create CSRR outer', vba_csrr_out);
        
        % 4.6 创建CSRR内环开口
        vba_csrr_in = [...
            'With Brick' newline ...
            '  .Reset' newline ...
            '  .Name "CSRR_Inner"' newline ...
            '  .Component "CSRR"' newline ...
            '  .Material "Vacuum"' newline ...
            sprintf('  .Xrange "%.4f", "%.4f"', -params.csrr.L_in/2, params.csrr.L_in/2) newline ...
            sprintf('  .Yrange "%.4f", "%.4f"', y_bottom, y_bottom + 0.035) newline ...
            sprintf('  .Zrange "%.4f", "%.4f"', params.csrr.z_pos - params.csrr.L_in/2, params.csrr.z_pos + params.csrr.L_in/2) newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'create CSRR inner', vba_csrr_in);
        
        % 布尔减法
        invoke(mws, 'AddToHistory', 'boolean subtract', ...
            'Solid.Subtract "CSRR:CSRR_Outer", "CSRR:CSRR_Inner"');
        fprintf('  ✓ CSRR结构创建完成\n');
        
        % 4.7 创建端口
        vba_port1 = [...
            'With Port' newline ...
            '  .Reset' newline ...
            '  .PortNumber "1"' newline ...
            '  .NumberOfModes "1"' newline ...
            '  .Orientation "zmin"' newline ...
            '  .Coordinates "Full"' newline ...
            sprintf('  .Xrange "%.4f", "%.4f"', -params.wg.a/2, params.wg.a/2) newline ...
            sprintf('  .Yrange "%.4f", "%.4f"', -params.wg.b/2, params.wg.b/2) newline ...
            '  .Zrange "0", "0"' newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'define port 1', vba_port1);
        
        vba_port2 = [...
            'With Port' newline ...
            '  .Reset' newline ...
            '  .PortNumber "2"' newline ...
            '  .NumberOfModes "1"' newline ...
            '  .Orientation "zmax"' newline ...
            '  .Coordinates "Full"' newline ...
            sprintf('  .Xrange "%.4f", "%.4f"', -params.wg.a/2, params.wg.a/2) newline ...
            sprintf('  .Yrange "%.4f", "%.4f"', -params.wg.b/2, params.wg.b/2) newline ...
            sprintf('  .Zrange "%.4f", "%.4f"', params.wg.length, params.wg.length) newline ...
            '  .Create' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'define port 2', vba_port2);
        fprintf('  ✓ 端口创建完成\n');
        
        % 4.8 设置边界条件
        vba_bc = [...
            'With Boundary' newline ...
            '  .Xmin "electric"' newline ...
            '  .Xmax "electric"' newline ...
            '  .Ymin "electric"' newline ...
            '  .Ymax "electric"' newline ...
            '  .Zmin "open"' newline ...
            '  .Zmax "open"' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'define boundary', vba_bc);
        fprintf('  ✓ 边界条件设置完成\n');
        
        % 4.9 配置时域求解器
        vba_td_solver = [...
            'With Solver' newline ...
            '  .Method "Hexahedral"' newline ...
            '  .CalculationType "TD-S"' newline ...
            '  .StimulationPort "1"' newline ...
            '  .SteadyStateLimit "-30"' newline ...
            '  .MeshAdaption "False"' newline ...
            '  .AutoNormImpedance "True"' newline ...
            '  .NormingImpedance "50"' newline ...
            '  .TimeBetweenUpdates "20"' newline ...
            'End With'];
        invoke(mws, 'AddToHistory', 'configure TD solver', vba_td_solver);
        fprintf('  ✓ 时域求解器配置完成\n');
        
        % 4.10 设置用户自定义激励
        % 将激励文件复制到CST项目目录
        excitation_file_cst = strrep(excitation_file, '\', '/');
        
        vba_excitation = [...
            'With Solver' newline ...
            '  .ExcitationPortMode "1", "1", "1", "True"' newline ...
            sprintf('  .ExcitationSignalAsReference "1", "%s", "1", "False"', excitation_file_cst) newline ...
            'End With'];
        
        % 备选方案：使用标准高斯激励（如果自定义激励失败）
        vba_excitation_default = [...
            'With Solver' newline ...
            '  .ExcitationPortMode "1", "1", "1", "True"' newline ...
            'End With'];
        
        try
            invoke(mws, 'AddToHistory', 'set excitation', vba_excitation);
            fprintf('  ✓ 用户自定义激励设置完成\n');
        catch
            invoke(mws, 'AddToHistory', 'set excitation', vba_excitation_default);
            fprintf('  ⚠ 使用默认高斯激励\n');
        end
        
        % 4.11 保存项目
        project_file = fullfile(data_dir, [params.cst.project_name '.cst']);
        invoke(mws, 'SaveAs', project_file, 'false');
        fprintf('  ✓ 项目已保存: %s\n', project_file);
        
        MODEL_CREATED = true;
        
    catch ME
        fprintf('  ✗ 模型创建失败: %s\n', ME.message);
        MODEL_CREATED = false;
    end
else
    MODEL_CREATED = false;
end

%% 5. 运行时域仿真
if CST_CONNECTED && MODEL_CREATED
    fprintf('\n【步骤4/6】运行时域仿真...\n');
    fprintf('  ⏳ 仿真进行中，请稍候...\n');
    
    try
        solver = invoke(mws, 'Solver');
        invoke(solver, 'Start');
        fprintf('  ✓ 仿真完成!\n');
        
        SIM_COMPLETED = true;
    catch ME
        fprintf('  ✗ 仿真失败: %s\n', ME.message);
        SIM_COMPLETED = false;
    end
else
    SIM_COMPLETED = false;
end

%% 6. 导出端口信号
if CST_CONNECTED && SIM_COMPLETED
    fprintf('\n【步骤5/6】导出端口信号...\n');
    
    try
        output_file = fullfile(data_dir, 'port2_signal.txt');
        output_file_cst = strrep(output_file, '\', '/');
        
        vba_export = [...
            'With ASCIIExport' newline ...
            '  .Reset' newline ...
            sprintf('  .FileName "%s"', output_file_cst) newline ...
            '  .Execute' newline ...
            'End With'];
        
        invoke(mws, 'AddToHistory', 'export signal', vba_export);
        fprintf('  ✓ 信号导出完成: %s\n', output_file);
        
        EXPORT_COMPLETED = true;
    catch ME
        fprintf('  ⚠ 信号导出失败: %s\n', ME.message);
        fprintf('    请手动导出Port 2信号\n');
        EXPORT_COMPLETED = false;
    end
else
    EXPORT_COMPLETED = false;
end

%% 7. 信号后处理
fprintf('\n【步骤6/6】信号后处理与参数反演...\n');

% 检查是否有CST输出数据
output_file = fullfile(data_dir, 'port2_signal.txt');

if exist(output_file, 'file')
    % 读取CST输出
    fprintf('  正在读取CST输出...\n');
    data = load(output_file);
    t_rx = data(:, 1) * 1e-9;  % ns -> s
    s_rx = data(:, 2);
    USE_REAL_DATA = true;
else
    % 使用模拟数据
    fprintf('  ⚠ 使用模拟数据进行演示...\n');
    
    % 模拟Lorentz介质的群时延效应
    % 简化模型：时延 + 频率相关衰减
    f_res = 34.5e9;   % 谐振频率
    gamma = 0.5e9;    % 阻尼
    tau_base = 0.3e-9;
    
    % 对每个时间点计算瞬时频率对应的群时延
    f_inst = params.lfmcw.f_start + params.lfmcw.K * t;
    tau_lorentz = tau_base + 0.5e-9 * gamma^2 ./ ((f_inst - f_res).^2 + gamma^2);
    
    % 可变时延（近似）
    delay_samples = round(mean(tau_lorentz) * params.lfmcw.f_s);
    s_rx = [zeros(1, delay_samples), s_tx(1:end-delay_samples)];
    s_rx = s_rx(:);
    t_rx = t(:);
    
    USE_REAL_DATA = false;
end

% 混频处理
fprintf('  正在进行混频处理...\n');
N_min = min(length(s_tx), length(s_rx));
s_tx_proc = s_tx(1:N_min);
s_rx_proc = s_rx(1:N_min);
t_proc = t(1:N_min);

s_mix = s_tx_proc(:) .* s_rx_proc(:);

% 低通滤波
fc_lp = 500e6;
[b, a] = butter(4, fc_lp/(params.lfmcw.f_s/2));
s_if = filtfilt(b, a, s_mix);

% FFT分析
N_fft = length(s_if);
S_IF = fft(s_if, N_fft);
S_IF_mag = abs(S_IF);
f_fft = (0:N_fft-1) * params.lfmcw.f_s / N_fft;

% 找峰值
[~, idx_peak] = max(S_IF_mag(1:round(N_fft/2)));
f_beat = f_fft(idx_peak);
tau_est = f_beat / params.lfmcw.K;

fprintf('  ✓ 差频峰值: %.3f kHz\n', f_beat/1e3);
fprintf('  ✓ 估计时延: %.3f ns\n', tau_est*1e9);

%% 8. 结果可视化
fprintf('\n正在生成可视化结果...\n');

figure('Position', [100, 100, 1200, 800]);

% 激励信号
subplot(2,3,1);
plot(t*1e6, s_tx);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('LFMCW激励信号', 'FontName', 'SimHei');
grid on; xlim([0, params.lfmcw.T_m*1e6]);

% 发射vs接收
subplot(2,3,2);
t_show = min(0.5e-6, params.lfmcw.T_m);
idx_show = t_proc < t_show;
plot(t_proc(idx_show)*1e9, s_tx_proc(idx_show), 'b', ...
     t_proc(idx_show)*1e9, s_rx_proc(idx_show), 'r');
xlabel('时间 (ns)', 'FontName', 'SimHei');
legend('发射', '接收');
title('发射 vs 接收 (局部)', 'FontName', 'SimHei');
grid on;

% 差频信号
subplot(2,3,3);
plot(t_proc*1e6, s_if);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('差频信号', 'FontName', 'SimHei');
grid on;

% 差频频谱
subplot(2,3,4);
f_lim = 10e6;
idx_f = f_fft < f_lim;
plot(f_fft(idx_f)/1e3, S_IF_mag(idx_f));
xlabel('频率 (kHz)', 'FontName', 'SimHei');
ylabel('幅度', 'FontName', 'SimHei');
title('差频频谱', 'FontName', 'SimHei');
grid on;
hold on;
xline(f_beat/1e3, 'r--', 'LineWidth', 2);

% 结果汇总
subplot(2,3,5);
text(0.5, 0.8, '【测试结果】', 'HorizontalAlignment', 'center', ...
    'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');
text(0.5, 0.6, sprintf('差频峰值: %.2f kHz', f_beat/1e3), ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
text(0.5, 0.4, sprintf('估计时延: %.3f ns', tau_est*1e9), ...
    'HorizontalAlignment', 'center', 'FontSize', 12);
if USE_REAL_DATA
    text(0.5, 0.2, '数据来源: CST仿真', 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'Color', 'g');
else
    text(0.5, 0.2, '数据来源: 模拟数据', 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'Color', [0.8, 0.4, 0]);
end
axis off;

% 状态
subplot(2,3,6);
status_text = {
    sprintf('CST连接: %s', bool2str(CST_CONNECTED));
    sprintf('模型创建: %s', bool2str(MODEL_CREATED));
    sprintf('仿真完成: %s', bool2str(SIM_COMPLETED));
    sprintf('数据导出: %s', bool2str(EXPORT_COMPLETED));
    sprintf('真实数据: %s', bool2str(USE_REAL_DATA));
};
for i = 1:length(status_text)
    text(0.1, 1 - i*0.15, status_text{i}, 'FontSize', 11, 'FontName', 'Consolas');
end
axis off;
title('执行状态', 'FontName', 'SimHei');

sgtitle('CST时域仿真全自动测试结果', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

%% 9. 保存结果
result_file = fullfile(data_dir, 'timedomain_results.mat');
save(result_file, 'params', 's_tx', 's_rx', 's_if', 'f_beat', 'tau_est', ...
    'CST_CONNECTED', 'MODEL_CREATED', 'SIM_COMPLETED', 'USE_REAL_DATA');
fprintf('\n✓ 结果已保存: %s\n', result_file);

%% 总结
fprintf('\n================================================\n');
fprintf('【执行总结】\n');
fprintf('================================================\n');
fprintf('CST连接: %s\n', bool2str(CST_CONNECTED));
fprintf('模型创建: %s\n', bool2str(MODEL_CREATED));
fprintf('仿真完成: %s\n', bool2str(SIM_COMPLETED));
fprintf('估计时延: %.3f ns\n', tau_est*1e9);
fprintf('================================================\n');
fprintf('\n✓ 全自动脚本执行完成!\n');

%% 辅助函数
function s = bool2str(b)
    if b
        s = '✓ 成功';
    else
        s = '✗ 失败';
    end
end
