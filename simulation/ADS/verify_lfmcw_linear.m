%% LFMCW 信号线性度验证脚本
% 输入文件：ads_data.txt (两列：时间, 电压)
% 功能：重采样 ADS 变步长数据，计算瞬时频率，验证线性度

clc; clear; close all;

%% 1. 全局字体设置
% 尝试设定中文字体，防止中文乱码
font_cn = 'SimSun'; % 宋体
try
    listfonts(font_cn);
catch
    font_cn = 'Microsoft YaHei'; % 微软雅黑
end
set(groot, 'defaultTextFontName', font_cn);
set(groot, 'defaultAxesFontName', font_cn);

%% 2. 数据导入与预处理
filename = 'ads_data.txt'; 

if ~isfile(filename)
    error('错误：找不到文件 %s，请确认文件在当前目录下。', filename);
end

fprintf('正在读取数据文件: %s ...\n', filename);

try
    % 读取数据，跳过可能存在的1行表头
    data = readmatrix(filename, 'NumHeaderLines', 1); 
    
    % 检查数据维度
    if size(data, 2) < 2
        error('数据列数不足，需要至少两列 (Time, Voltage)。');
    end
    
    t_raw = data(:, 1);
    v_raw = data(:, 2);
catch ME
    warning('readmatrix 读取失败，尝试 importdata ...');
    try
        raw_struct = importdata(filename);
        data = raw_struct.data;
        t_raw = data(:, 1);
        v_raw = data(:, 2);
    catch
        error('无法读取文件。请确保格式为两列数值（可带表头）。\n错误信息: %s', ME.message);
    end
end

% --- 去除重复时间点 ---
% ADS 瞬态仿真有时会输出重复时间点，导致插值失败
[t_raw, unique_idx] = unique(t_raw);
v_raw = v_raw(unique_idx);

%% 3. 关键步骤：重采样 (Resampling)
% ADS 输出是变步长 (Variable Time Step) 的，
% 必须插值为固定采样率才能进行 FFT 或 Hilbert 变换。

% 设置目标采样率
% 信号最高频率约 38 GHz，根据采样定理 Fs > 2*Fmax。
% 为了获得平滑的波形和高精度相位，设 Fs = 100 GHz (10 ps)
Fs = 100e9; 
dt = 1/Fs;

fprintf('正在进行重采样 (Target Fs = %.1f GHz)...\n', Fs/1e9);

% 创建均匀时间轴
t_uniform = (min(t_raw) : dt : max(t_raw))';

% 使用线性插值将数据映射到新时间轴
v_uniform = interp1(t_raw, v_raw, t_uniform, 'linear');

% 去除直流分量 (DC Offset)
v_uniform = v_uniform - mean(v_uniform);

fprintf('数据重采样完成。点数: %d\n', length(v_uniform));

%% 4. 方法一：时频图验证 (Spectrogram) - 论文级可视化优化版
figure('Name', 'LFMCW 时频分析', 'Color', 'w', 'Position', [100, 400, 800, 500]);

% STFT 参数设置
window_size = 512;
overlap = 256;
nfft = 1024;

% 计算 STFT 数据
[s, f_spec, t_spec] = spectrogram(v_uniform, window_size, overlap, nfft, Fs);

% --- 关键修改：手动绘图以控制坐标轴和颜色 ---
% 1. 将时间转换为 ns，频率转换为 GHz
% 2. 使用 imagesc 绘图
imagesc(t_spec*1e9, f_spec/1e9, 20*log10(abs(s))); 
axis xy; % 翻转 Y 轴，让低频在下，高频在上

% --- 美化设置 ---
title('方法一：短时傅里叶变换 (STFT) 时频图', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', font_cn);
ylabel('频率 (GHz)', 'FontSize', 11, 'FontName', font_cn);
xlabel('时间 (ns)', 'FontSize', 11, 'FontName', font_cn);
colormap('jet'); % 使用彩虹色谱
cb = colorbar;
cb.Label.String = '功率谱密度 (dB)';
cb.Label.FontName = font_cn;

% --- 动态调整显示范围 (重点) ---
% 这里的 ylim 限制在 30-40G，让信号填满画面
ylim([30 40]); 
% 这里的 clim (或 caxis) 限制颜色范围，只显示最强的信号部分 (Top 40dB)
max_pwr = max(max(20*log10(abs(s))));
try
    clim([max_pwr-40, max_pwr]); 
catch
    caxis([max_pwr-40, max_pwr]); % 兼容旧版本 MATLAB
end 

grid on;
set(gca, 'FontSize', 10, 'FontName', font_cn); 

%% 5. 方法二：希尔伯特变换瞬时频率 (Numerical Proof)
% 提取瞬时频率，计算线性度 R^2。

fprintf('正在计算瞬时频率 (Hilbert Transform)...\n');

% --- 计算解析信号 ---
z = hilbert(v_uniform);

% --- 提取瞬时相位 ---
% unwrap 处理相位卷绕 (2pi 跳变)
inst_phase = unwrap(angle(z));

% --- 求导得到瞬时频率 (f = d(phi)/dt / 2pi) ---
% diff 计算相邻点差分
inst_freq = diff(inst_phase) / (2*pi*dt);

% 频率对应的时间轴（长度比原始少 1）
t_freq = t_uniform(1:end-1);

% --- 线性拟合验证 ---
% 为了避开首尾的吉布斯效应（边缘震荡），只取中间 80% 的稳定数据段做验证
idx_start = floor(length(t_freq) * 0.1);
idx_end = floor(length(t_freq) * 0.9);

t_fit = t_freq(idx_start:idx_end);
f_fit = inst_freq(idx_start:idx_end);

% 对数据进行平滑处理（可选，减少数值微分噪声）
% f_fit = smoothdata(f_fit, 'movmean', 5);

% 多项式拟合 (1阶 = 直线 y = ax + b)
[p, S] = polyfit(t_fit, f_fit, 1); 
f_linear_model = polyval(p, t_fit);

% 计算拟合优度 R-squared
y_resid = f_fit - f_linear_model;
SSresid = sum(y_resid.^2);
SStotal = (length(f_fit)-1) * var(f_fit);
rsq = 1 - SSresid/SStotal;

%% 6. 绘制瞬时频率结果
figure('Name', '瞬时频率线性度验证', 'Color', 'w', 'Position', [150, 150, 800, 500]);

% 绘制提取的频率曲线
plot(t_freq*1e9, inst_freq/1e9, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, ...
    'DisplayName', '提取的瞬时频率'); 
hold on;

% 绘制理想线性拟合
plot(t_fit*1e9, f_linear_model/1e9, 'b--', 'LineWidth', 2, ...
    'DisplayName', '理想线性拟合');

title({['方法二：希尔伯特变换瞬时频率']; ...
       ['线性度 R^2 = ' num2str(rsq, '%.6f') ' (1.0 为完美线性)']}, ...
       'FontSize', 12);
xlabel('时间 (ns)', 'FontSize', 11);
ylabel('频率 (GHz)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 10);
grid on;

% 聚焦显示范围
ylim([30 40]); 

% 输出验证报告
chirp_rate = p(1); % Hz/s
fprintf('\n------------------------------------------------\n');
fprintf('LFMCW 线性度验证结果:\n');
fprintf('------------------------------------------------\n');
fprintf('线性拟合优度 (R^2): %.6f\n', rsq);
if rsq > 0.99
    fprintf('结论: 信号具有极高的线性度，确认为 LFMCW 信号。\n');
else
    fprintf('结论: 线性度较低，可能存在非线性或噪声干扰。\n');
end
fprintf('调频斜率 (Slope): %.2e Hz/s\n', chirp_rate);
fprintf('起始频率 (Fit):   %.2f GHz\n', polyval(p, t_fit(1))/1e9);
fprintf('终止频率 (Fit):   %.2f GHz\n', polyval(p, t_fit(end))/1e9);
fprintf('------------------------------------------------\n');
