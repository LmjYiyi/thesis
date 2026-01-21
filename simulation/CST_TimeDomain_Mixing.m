%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CST时域数据混频处理与参数反演
% 功能: 读取CST时域仿真数据（发射i1、接收o21），进行混频处理得到差频信号
%       参考 LM_MCMC.m 的处理流程
%
% 数据来源: CST Microwave Studio 时域仿真
% 作者: Auto-generated for thesis project
% 日期: 2026-01-20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 初始化
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end

fprintf('================================================\n');
fprintf('CST时域数据混频处理与参数反演\n');
fprintf('================================================\n\n');

%% 2. 读取CST数据
fprintf('【步骤1】读取CST时域数据...\n');

data_file = fullfile(script_dir, 'cst_data', 'output.txt');

if ~exist(data_file, 'file')
    error('找不到数据文件: %s', data_file);
end

% 读取整个文件
fid = fopen(data_file, 'r');
raw_text = fread(fid, '*char')';
fclose(fid);

% 分割为两个数据块
% i1数据从第4行开始，o21数据从"o2,1"标记后开始
lines = strsplit(raw_text, {'\r\n', '\n'});

% 找到o21数据的起始位置
o21_start_idx = 0;
for i = 1:length(lines)
    if contains(lines{i}, 'o2,1')
        o21_start_idx = i + 2;  % 跳过标题行和分隔线
        break;
    end
end

if o21_start_idx == 0
    error('未找到o21数据标记');
end

% 解析i1数据 (第4行到o21标记前)
i1_lines = lines(4:o21_start_idx-4);
i1_data = zeros(length(i1_lines), 2);
valid_count = 0;
for i = 1:length(i1_lines)
    vals = sscanf(i1_lines{i}, '%f\t%f');
    if length(vals) == 2
        valid_count = valid_count + 1;
        i1_data(valid_count, :) = vals';
    end
end
i1_data = i1_data(1:valid_count, :);

% 解析o21数据
o21_lines = lines(o21_start_idx:end);
o21_data = zeros(length(o21_lines), 2);
valid_count = 0;
for i = 1:length(o21_lines)
    vals = sscanf(o21_lines{i}, '%f\t%f');
    if length(vals) == 2
        valid_count = valid_count + 1;
        o21_data(valid_count, :) = vals';
    end
end
o21_data = o21_data(1:valid_count, :);

fprintf('  ✓ i1 数据: %d 个采样点\n', size(i1_data, 1));
fprintf('  ✓ o21 数据: %d 个采样点\n', size(o21_data, 1));

% 提取时间和信号
t_i1 = i1_data(:, 1) * 1e-9;   % ns -> s
s_tx = i1_data(:, 2);

t_o21 = o21_data(:, 1) * 1e-9;  % ns -> s
s_rx = o21_data(:, 2);

% 计算采样率
dt = mean(diff(t_i1));
f_s = 1 / dt;
fprintf('  采样率: %.3f GHz\n', f_s/1e9);
fprintf('  时间范围: %.2f - %.2f ns\n', t_i1(1)*1e9, t_i1(end)*1e9);

%% 3. 数据预处理
fprintf('\n【步骤2】数据预处理...\n');

% 确保两个信号长度相同
N_min = min(length(s_tx), length(s_rx));
s_tx = s_tx(1:N_min);
s_rx = s_rx(1:N_min);
t = t_i1(1:N_min);
N = length(t);

fprintf('  有效数据点: %d\n', N);

%% 4. 混频处理
fprintf('\n【步骤3】混频处理...\n');

% 混频：发射信号 × 接收信号
s_mix = s_tx .* s_rx;

% 低通滤波器设计
fc_lp = 2e9;  % 低通截止频率 2 GHz
[b_lp, a_lp] = butter(4, fc_lp / (f_s/2));

% 滤波
s_if = filtfilt(b_lp, a_lp, s_mix);

fprintf('  ✓ 混频完成\n');
fprintf('  ✓ 低通滤波完成 (fc = %.1f GHz)\n', fc_lp/1e9);

%% 5. 差频信号分析
fprintf('\n【步骤4】差频信号分析...\n');

% 加窗
win = hann(N);
s_if_win = s_if .* win;

% FFT
S_IF = fft(s_if_win);
S_IF_mag = abs(S_IF) * 2 / N;

% 频率轴
f_fft = (0:N-1) * f_s / N;

% 找差频峰值（只看正频率部分）
f_max_search = 50e9;  % 最大搜索频率
idx_search = f_fft <= f_max_search;
[peak_val, peak_idx] = max(S_IF_mag(idx_search));
f_beat = f_fft(peak_idx);

fprintf('  差频峰值频率: %.3f GHz\n', f_beat/1e9);
fprintf('  差频峰值幅度: %.4f\n', peak_val);

%% 6. LFMCW参数估算
fprintf('\n【步骤5】LFMCW参数估算...\n');

% 从信号分析LFMCW参数
% 估计LFMCW参数（从图像可知约34-37 GHz范围）
% 假设已知LFMCW参数：
f_start_est = 34.2e9;   % 起始频率
f_end_est = 37.4e9;     % 终止频率
T_m_est = 50e-9;        % 扫频周期（从数据看约50ns）
B_est = f_end_est - f_start_est;
K_est = B_est / T_m_est;  % 调频斜率

fprintf('  估计起始频率: %.2f GHz\n', f_start_est/1e9);
fprintf('  估计终止频率: %.2f GHz\n', f_end_est/1e9);
fprintf('  估计扫频周期: %.1f ns\n', T_m_est*1e9);
fprintf('  估计调频斜率: %.3f GHz/ns\n', K_est/1e18);

% 从差频估计时延
tau_est = f_beat / K_est;
fprintf('\n  ★ 估计群时延: %.3f ns\n', tau_est*1e9);

%% 7. 可视化
fprintf('\n【步骤6】生成可视化结果...\n');

figure('Position', [100, 100, 1400, 900]);

% 7.1 发射信号
subplot(3,3,1);
t_display = min(5e-9, t(end));  % 显示前5ns
idx_disp = t <= t_display;
plot(t(idx_disp)*1e9, s_tx(idx_disp), 'b', 'LineWidth', 0.5);
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('发射信号 i1 (局部)', 'FontName', 'SimHei');
grid on;

% 7.2 接收信号
subplot(3,3,2);
plot(t(idx_disp)*1e9, s_rx(idx_disp), 'r', 'LineWidth', 0.5);
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('接收信号 o21 (局部)', 'FontName', 'SimHei');
grid on;

% 7.3 发射vs接收对比
subplot(3,3,3);
idx_compare = t <= 1e-9;  % 前1ns对比
plot(t(idx_compare)*1e9, s_tx(idx_compare), 'b', ...
     t(idx_compare)*1e9, s_rx(idx_compare), 'r--', 'LineWidth', 1);
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('发射 vs 接收 (前1ns)', 'FontName', 'SimHei');
legend('发射', '接收');
grid on;

% 7.4 混频信号
subplot(3,3,4);
plot(t*1e9, s_mix, 'Color', [0.3, 0.3, 0.3], 'LineWidth', 0.3);
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('混频信号 (原始)', 'FontName', 'SimHei');
grid on;

% 7.5 差频信号
subplot(3,3,5);
plot(t*1e9, s_if, 'b', 'LineWidth', 0.5);
xlabel('时间 (ns)', 'FontName', 'SimHei');
ylabel('幅值', 'FontName', 'SimHei');
title('差频信号 (低通滤波后)', 'FontName', 'SimHei');
grid on;

% 7.6 差频频谱
subplot(3,3,6);
f_lim = 20e9;  % 显示频率范围
idx_f = f_fft <= f_lim;
plot(f_fft(idx_f)/1e9, S_IF_mag(idx_f), 'b', 'LineWidth', 1);
hold on;
xline(f_beat/1e9, 'r--', 'LineWidth', 2);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅度', 'FontName', 'SimHei');
title(sprintf('差频频谱 (峰值: %.2f GHz)', f_beat/1e9), 'FontName', 'SimHei');
grid on;

% 7.7 发射信号频谱
subplot(3,3,7);
S_TX = fft(s_tx .* win);
S_TX_mag = abs(S_TX) * 2 / N;
f_range_idx = (f_fft >= 30e9) & (f_fft <= 40e9);
plot(f_fft(f_range_idx)/1e9, S_TX_mag(f_range_idx), 'b', 'LineWidth', 1);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅度', 'FontName', 'SimHei');
title('发射信号频谱 (30-40 GHz)', 'FontName', 'SimHei');
grid on;

% 7.8 接收信号频谱
subplot(3,3,8);
S_RX = fft(s_rx .* win);
S_RX_mag = abs(S_RX) * 2 / N;
plot(f_fft(f_range_idx)/1e9, S_RX_mag(f_range_idx), 'r', 'LineWidth', 1);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('幅度', 'FontName', 'SimHei');
title('接收信号频谱 (30-40 GHz)', 'FontName', 'SimHei');
grid on;

% 7.9 结果汇总
subplot(3,3,9);
result_text = {
    sprintf('采样率: %.2f GHz', f_s/1e9);
    sprintf('数据点数: %d', N);
    sprintf('时间范围: 0 - %.1f ns', t(end)*1e9);
    '';
    sprintf('差频峰值: %.3f GHz', f_beat/1e9);
    sprintf('估计时延: %.3f ns', tau_est*1e9);
    '';
    sprintf('调频斜率: %.2f GHz/ns', K_est/1e18);
};
for i = 1:length(result_text)
    text(0.1, 1 - i*0.1, result_text{i}, 'FontSize', 11, 'FontName', 'Consolas');
end
axis off;
title('分析结果汇总', 'FontName', 'SimHei');

sgtitle('CST时域仿真数据混频处理', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

%% 8. 保存结果
result_file = fullfile(script_dir, 'mixing_results.mat');
save(result_file, 't', 's_tx', 's_rx', 's_if', 'f_beat', 'tau_est', 'f_s', 'K_est');
fprintf('\n✓ 结果已保存: %s\n', result_file);

% 保存图像
fig_file = fullfile(script_dir, 'mixing_results.png');
print(gcf, '-dpng', '-r150', fig_file);
fprintf('✓ 图像已保存: %s\n', fig_file);

%% 9. 总结
fprintf('\n================================================\n');
fprintf('【处理完成】\n');
fprintf('================================================\n');
fprintf('差频峰值频率: %.3f GHz\n', f_beat/1e9);
fprintf('估计群时延: %.3f ns\n', tau_est*1e9);
fprintf('================================================\n');
