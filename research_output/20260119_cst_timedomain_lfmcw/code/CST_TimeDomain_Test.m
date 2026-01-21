%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST 时域仿真测试脚本 - 简化版
% 功能: 生成LFMCW激励信号 → CST时域仿真 → 读取输出 → 混频反演
%
% 使用方法:
%   1. 先运行第1节生成激励信号文件
%   2. 在CST中手动加载激励并运行时域仿真
%   3. 导出端口2信号到 data/timedomain_output.txt
%   4. 运行第2节进行后处理
%
% 作者: Auto-generated for thesis project
% 日期: 2026-01-19
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 0. 路径设置
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
data_dir = fullfile(script_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

%% =========================================================================
%  第1节: 生成LFMCW激励信号
%  =========================================================================

fprintf('========================================\n');
fprintf('CST 时域仿真测试脚本\n');
fprintf('========================================\n\n');

% -----------------------
% 1.1 LFMCW参数设置
% -----------------------
% 注意: 为减少CST计算时间，使用缩短的扫频周期
f_start = 34.2e9;       % 起始频率 (Hz)
f_end = 37.4e9;         % 终止频率 (Hz)
T_m = 5e-6;             % 扫频周期 (s) - 缩短为5us以节省仿真时间!
f_s = 100e9;            % CST采样率 (Hz)

% 派生参数
B = f_end - f_start;    % 带宽
K = B / T_m;            % 调频斜率
c = 3e8;

fprintf('【第1节】生成LFMCW激励信号\n');
fprintf('-----------------------\n');
fprintf('频率范围: %.2f - %.2f GHz\n', f_start/1e9, f_end/1e9);
fprintf('扫频周期: %.2f us (缩短以节省仿真时间)\n', T_m*1e6);
fprintf('采样率: %.1f GHz\n', f_s/1e9);
fprintf('调频斜率: %.2f GHz/us\n', K/1e15);

% -----------------------
% 1.2 生成信号
% -----------------------
t = 0 : 1/f_s : T_m;
N = length(t);

phi = 2*pi*f_start*t + pi*K*t.^2;
s_tx = cos(phi);

fprintf('采样点数: %d\n', N);

% -----------------------
% 1.3 保存为CST格式
% -----------------------
excitation_file = fullfile(data_dir, 'lfmcw_excitation.sig');

fid = fopen(excitation_file, 'w');
fprintf(fid, '%% LFMCW Excitation for CST Time Domain Solver\n');
fprintf(fid, '%% time (ns)  amplitude\n');
for i = 1:N
    fprintf(fid, '%.9e  %.9e\n', t(i)*1e9, s_tx(i));
end
fclose(fid);

fprintf('\n✓ 激励信号已保存: %s\n', excitation_file);
fprintf('  文件大小: %.2f MB\n', dir(excitation_file).bytes/1e6);

% -----------------------
% 1.4 可视化激励信号
% -----------------------
figure(1);
subplot(2,1,1);
plot(t*1e6, s_tx);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('LFMCW激励信号 (时域)', 'FontName', 'SimHei');
grid on; xlim([0, T_m*1e6]);

subplot(2,1,2);
f_inst = f_start + K*t;
plot(t*1e6, f_inst/1e9);
xlabel('时间 (μs)', 'FontName', 'SimHei');
ylabel('瞬时频率 (GHz)', 'FontName', 'SimHei');
title('瞬时频率', 'FontName', 'SimHei');
grid on; xlim([0, T_m*1e6]);

sgtitle('LFMCW激励信号', 'FontName', 'SimHei', 'FontWeight', 'bold');

%% CST操作指南 (需要手动执行)
fprintf('\n========================================\n');
fprintf('【CST操作指南】请手动执行以下步骤:\n');
fprintf('========================================\n');
fprintf('1. 打开CST项目 (CSRR_WR28_Lorentz.cst)\n');
fprintf('2. 切换到时域求解器:\n');
fprintf('   → Simulation → Setup Solver → Time Domain\n');
fprintf('3. 设置用户自定义激励:\n');
fprintf('   → 在Port设置中选择 "User Defined"\n');
fprintf('   → 加载文件: %s\n', excitation_file);
fprintf('4. 运行仿真\n');
fprintf('5. 导出Port 2输出信号:\n');
fprintf('   → Post Processing → Port Signals → Export\n');
fprintf('   → 保存为: %s\\timedomain_output.txt\n', data_dir);
fprintf('========================================\n');
fprintf('\n完成CST仿真后，请运行下一节...\n');

%% =========================================================================
%  第2节: 读取CST输出并后处理
%  =========================================================================
% 取消下面的 "return" 注释来分步执行
% return;  

fprintf('\n【第2节】读取CST输出并后处理\n');
fprintf('-----------------------\n');

% -----------------------
% 2.1 读取CST输出
% -----------------------
output_file = fullfile(data_dir, 'timedomain_output.txt');

if exist(output_file, 'file')
    fprintf('正在读取CST输出...\n');
    
    % 读取数据 (假设格式: time amplitude)
    data = load(output_file);
    t_rx = data(:, 1) * 1e-9;  % ns -> s
    s_rx = data(:, 2);
    
    fprintf('✓ 读取完成: %d 个采样点\n', length(s_rx));
    
else
    % 如果没有CST数据，使用模拟数据进行演示
    fprintf('⚠ 未找到CST输出文件，使用模拟数据演示...\n');
    
    % 模拟通过Lorentz介质的传播
    tau_air = 0.5e-9;  % 模拟空气时延
    
    % 简单时延 + 幅度调制
    delay_samples = round(tau_air * f_s);
    s_rx = [zeros(1, delay_samples), s_tx(1:end-delay_samples)];
    s_rx = s_rx(:);
    t_rx = t(:);
    
    fprintf('  (使用简单时延模型模拟)\n');
end

% -----------------------
% 2.2 混频处理
% -----------------------
fprintf('正在进行混频处理...\n');

% 确保长度匹配
N_min = min(length(s_tx), length(s_rx));
s_tx_proc = s_tx(1:N_min);
s_rx_proc = s_rx(1:N_min);
t_proc = t(1:N_min);

% 混频
s_mix = s_tx_proc(:) .* s_rx_proc(:);

% 低通滤波
fc_lp = 500e6;  % 低通截止频率
[b, a] = butter(4, fc_lp/(f_s/2));
s_if = filtfilt(b, a, s_mix);

fprintf('✓ 混频完成\n');

% -----------------------
% 2.3 差频信号分析
% -----------------------
fprintf('正在分析差频信号...\n');

% FFT
N_fft = length(s_if);
S_IF = fft(s_if, N_fft);
S_IF_mag = abs(S_IF);
f_fft = (0:N_fft-1) * f_s / N_fft;

% 找峰值
[~, idx_peak] = max(S_IF_mag(1:round(N_fft/2)));
f_beat = f_fft(idx_peak);
tau_est = f_beat / K;

fprintf('✓ 差频峰值: %.3f kHz\n', f_beat/1e3);
fprintf('✓ 估计时延: %.3f ns\n', tau_est*1e9);

% -----------------------
% 2.4 结果可视化
% -----------------------
figure(2);
subplot(2,2,1);
plot(t_proc*1e6, s_tx_proc, 'b', t_proc*1e6, s_rx_proc, 'r');
xlabel('时间 (μs)', 'FontName', 'SimHei');
legend('发射', '接收');
title('发射 vs 接收信号', 'FontName', 'SimHei');
grid on; xlim([0, min(1, T_m*1e6)]);

subplot(2,2,2);
plot(t_proc*1e6, s_if);
xlabel('时间 (μs)', 'FontName', 'SimHei');
title('差频信号 (时域)', 'FontName', 'SimHei');
grid on;

subplot(2,2,3);
f_lim = 10e6;  % 显示范围
idx_show = f_fft < f_lim;
plot(f_fft(idx_show)/1e3, S_IF_mag(idx_show));
xlabel('频率 (kHz)', 'FontName', 'SimHei');
title('差频信号 (频谱)', 'FontName', 'SimHei');
grid on;

subplot(2,2,4);
text(0.5, 0.6, sprintf('差频峰值: %.2f kHz', f_beat/1e3), ...
    'HorizontalAlignment', 'center', 'FontSize', 14);
text(0.5, 0.4, sprintf('估计时延: %.3f ns', tau_est*1e9), ...
    'HorizontalAlignment', 'center', 'FontSize', 14);
axis off;
title('测量结果', 'FontName', 'SimHei');

sgtitle('CST时域仿真后处理结果', 'FontName', 'SimHei', 'FontWeight', 'bold');

%% 3. 总结
fprintf('\n========================================\n');
fprintf('【测试总结】\n');
fprintf('========================================\n');
fprintf('激励信号: %.2f - %.2f GHz, %.2f us\n', f_start/1e9, f_end/1e9, T_m*1e6);
fprintf('差频峰值: %.3f kHz\n', f_beat/1e3);
fprintf('估计时延: %.3f ns\n', tau_est*1e9);
fprintf('========================================\n');

fprintf('\n✓ 测试脚本执行完成!\n');
