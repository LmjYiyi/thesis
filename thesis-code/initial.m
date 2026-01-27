%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW等离子体诊断系统MATLAB仿真 - 净版 (无估算逻辑)
% 功能：生成信号、Drude模型传播、混频、绘制基础波形与频谱
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. 仿真参数设置 (修正频率轴逻辑)
clc; clear all; close all;

% LFMCW雷达参数
f_start = 34.2e9;            
f_end = 37.4e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

% 传播介质参数
tau_air = 4e-9;              
tau_fs = 1.75e-9;            
d = 150e-3;                  
f_c = 33e9;                  
nu = 1.5e9;                  

% 计算派生参数 (修正 n_e 定义防止报错)
c = 3e8;                     
omega_p = 2*pi*f_c;          
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;
n_e = (omega_p^2 * epsilon_0 * m_e) / e^2; % 显式计算 n_e

t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% --- 关键修正：构建正确的FFT频率轴 (包含负频率) ---
f = (0:N-1)*(f_s/N);         
% 将大于 fs/2 的频率映射为负频率
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

%% 2. LFMCW信号生成模块 (保持不变)
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              

%% 3. 信号传播模拟模块 (修正符号约定版)

% 3.1 空气介质传播模拟
delay_samples_air = round(tau_air/t_s);
s_rx_air = [zeros(1, delay_samples_air) s_tx(1:end-delay_samples_air)];

% 3.2 等离子体介质传播模拟 (频域处理)
% 步骤1: 第一段自由空间
delay_samples_fs = round(tau_fs/t_s);
s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];

% 步骤2: 进入频域
S_after_fs1 = fft(s_after_fs1);

% --- 核心物理计算 (向量化) ---
% 避免除零
omega_safe = omega; 
omega_safe(omega_safe == 0) = 1e-10; 

% Drude模型: 复介电常数
epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
epsilon_r_complex(omega == 0) = 1; 

% 复波数 k
k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);

% 传递函数 H (关键修正：强制衰减)
% 分解实部(相位)和虚部(衰减)
k_real = real(k_complex);
k_imag = imag(k_complex);

% 构造传递函数：
% 相位项: -1i * k_real * d
% 衰减项: -abs(k_imag) * d  <-- 必须是负数，确保波幅减小
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

% 频域施加影响
S_after_plasma = S_after_fs1 .* H_plasma;
S_RX_plasma_fft = S_after_plasma; 

% 步骤3: 回到时域 (取实部)
s_after_plasma = real(ifft(S_after_plasma));

% 步骤4: 第二段自由空间
s_rx_plasma = [zeros(1, delay_samples_fs) s_after_plasma(1:end-delay_samples_fs)];

%% 4. 混频处理与差频信号提取

% 4.1 空气介质混频
s_mix_air = s_tx .* s_rx_air;

% 低通滤波器设计 (提取差频)
fc_lp = 100e6;  % 截止频率100MHz (足以覆盖差频)
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));

s_if_air = filtfilt(b_lp, a_lp, s_mix_air);

% 4.2 等离子体介质混频
s_mix_plasma = s_tx .* real(s_rx_plasma);
s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);

%% 5. 基础频域分析 (FFT) - 仅用于绘图数据准备

% 5.1 空气介质差频 FFT
S_IF_air = fft(s_if_air, N);
S_IF_air_mag = abs(S_IF_air);

% 计算理论空气差频 (仅画图参考用)
f_beat_air_theory = K * tau_air;

% 5.2 等离子体介质差频 FFT (加汉宁窗抑制旁瓣)
win = hann(N)';
s_if_plasma_win = s_if_plasma .* win;
S_IF_plasma = fft(s_if_plasma_win, N);
S_IF_plasma_mag = abs(S_IF_plasma) * 2; % 补偿窗函数幅度损失

%% 6. 可视化 (Figure 1 - 8) - 严格保持原始波形与频谱

% -------------------------------------------------------------------------
% 1. 绘图数据准备 (直接读取 Section 5 的结果)
% -------------------------------------------------------------------------
% 定义正半轴频率向量 (用于绘图)
L_half = ceil(N/2);
f_axis_plot = (0:L_half-1) * (f_s / N); 

% 截取正频率部分的幅度 (直接使用第5节计算好的变量)
mag_air_plot = S_IF_air_mag(1:L_half);      % 空气 (无窗/矩形窗)
mag_plasma_plot = S_IF_plasma_mag(1:L_half); % 等离子体 (汉宁窗)

% -------------------------------------------------------------------------
% 2. Figure 1: 发射 vs 空气接收 (合并时域与频域)
% -------------------------------------------------------------------------
t_display = min(5e-6, T_m); idx_display = round(t_display/t_s);
f_range = [f_start-0.5e9, f_end+0.5e9]; f_indices = find(f >= f_range(1) & f <= f_range(2));

figure(1);
set(gcf, 'Position', [100, 100, 1200, 500]);

% Left: Time Domain
subplot(1, 2, 1);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, s_rx_air(1:idx_display), 'r--');
xlabel('时间 (μs)'); ylabel('幅值'); title('发射信号 vs 空气接收 (时域)'); legend('Tx', 'Rx Air'); grid on;

% Right: Frequency Domain
subplot(1, 2, 2);
S_TX_mag = abs(fft(s_tx)); S_RX_air_mag_orig = abs(fft(s_rx_air));
if ~isempty(f_indices), plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_air_mag_orig(f_indices), 'r--'); end
xlabel('频率 (GHz)'); title('发射信号 vs 空气接收 (频域)'); grid on;

% -------------------------------------------------------------------------
% 3. Figure 2: 发射 vs 等离子体接收 (合并时域与频域)
% -------------------------------------------------------------------------
figure(2);
set(gcf, 'Position', [150, 150, 1200, 500]);

% Left: Time Domain
subplot(1, 2, 1);
plot(t(1:idx_display)*1e6, s_tx(1:idx_display), 'b', t(1:idx_display)*1e6, real(s_rx_plasma(1:idx_display)), 'r--');
xlabel('时间 (μs)'); title('发射信号 vs 等离子体接收 (时域)'); grid on;

% Right: Frequency Domain
subplot(1, 2, 2);
S_RX_plasma_mag_orig = abs(S_RX_plasma_fft);
if ~isempty(f_indices), plot(f(f_indices)/1e9, S_TX_mag(f_indices), 'b', f(f_indices)/1e9, S_RX_plasma_mag_orig(f_indices), 'r--'); end
xlabel('频率 (GHz)'); title('发射信号 vs 等离子体接收 (频域)'); grid on;

% -------------------------------------------------------------------------
% 4. Figure 3: 差频信号分析 (合并 2x2)
% -------------------------------------------------------------------------
figure(3);
set(gcf, 'Position', [200, 200, 1200, 800]);

% 寻找显示范围
[~, k_peak_air_rough] = max(mag_air_plot);
[~, k_peak_pla_rough] = max(mag_plasma_plot);
f_peak_air_rough = f_axis_plot(k_peak_air_rough);
f_peak_pla_rough = f_axis_plot(k_peak_pla_rough);
zoom_span_vis = 0.8e6; 

% Top-Left: Air Time
subplot(2, 2, 1);
plot(t(1:round(20e-6/t_s))*1e6, s_if_air(1:round(20e-6/t_s)), 'b');
xlabel('时间 (μs)'); title('空气差频 (时域)'); grid on;

% Top-Right: Plasma Time
subplot(2, 2, 2);
plot(t(1:round(20e-6/t_s))*1e6, s_if_plasma(1:round(20e-6/t_s)), 'r');
xlabel('时间 (μs)'); title('等离子体差频 (时域)'); grid on;

% Bottom-Left: Air Freq (保持原样：很尖锐)
subplot(2, 2, 3);
stem(f_axis_plot/1e3, mag_air_plot, 'b', 'MarkerSize', 4, 'LineWidth', 1.0, 'BaseValue', 0); 
hold on;
xline(f_beat_air_theory/1e3, 'k--', 'LineWidth', 1.5, 'DisplayName', '理论真值');
xlim([(f_peak_air_rough - zoom_span_vis)/1e3, (f_peak_air_rough + zoom_span_vis)/1e3]);
xlabel('频率 (kHz)'); ylabel('幅值'); 
title('空气差频 (离散频谱)'); 
grid on; legend('离散频谱数据', '理论真值');

% Bottom-Right: Plasma Freq (保持原样：稍微胖一点)
subplot(2, 2, 4);
stem(f_axis_plot/1e3, mag_plasma_plot, 'b', 'MarkerSize', 4, 'LineWidth', 1.0, 'BaseValue', 0); 
xlim([(f_peak_pla_rough - zoom_span_vis)/1e3, (f_peak_pla_rough + zoom_span_vis)/1e3]);
xlabel('频率 (kHz)'); ylabel('幅值'); 
title('等离子体差频 (离散频谱)'); 
grid on; legend('离散频谱数据');


%% 7. 【传统方法诊断总结】Figure 10: 高精度校正分析
% 目的：为了保证计算精度，在此处对空气信号补加汉宁窗，确保与等离子体信号处理一致

% =========================================================================
% 7.1 数据再处理 (核心修正：统一加窗以恢复精度)
% =========================================================================
win_calc = hann(N)'; % 定义分析用的窗函数

% [空气数据重算] - 为了计算准，必须加窗
S_air_calc = fft(s_if_air .* win_calc, N);
mag_air_analysis = abs(S_air_calc(1:length(f_axis_plot)));

% [等离子体数据] - 沿用第5节已加窗的数据
mag_pla_analysis = mag_plasma_plot; 

% =========================================================================
% 7.2 执行三角形校正算法
% =========================================================================

% --- 空气信号校正 (基于 mag_air_analysis) ---
[val_peak_air, idx_air] = max(mag_air_analysis);
if idx_air > 1 && idx_air < length(mag_air_analysis)
    A_L = mag_air_analysis(idx_air - 1); 
    A_C = mag_air_analysis(idx_air); 
    A_R = mag_air_analysis(idx_air + 1);
    delta_k_air = (A_R - A_L) / (A_L + A_C + A_R);
    f_corr_air = (idx_air - 1 + delta_k_air) * (f_s / N);
else
    f_corr_air = f_axis_plot(idx_air);
end

% --- 等离子体信号校正 (基于 mag_pla_analysis) ---
[val_peak_pla, idx_pla] = max(mag_pla_analysis);
if idx_pla > 1 && idx_pla < length(mag_pla_analysis)
    A_L = mag_pla_analysis(idx_pla - 1); 
    A_C = mag_pla_analysis(idx_pla); 
    A_R = mag_pla_analysis(idx_pla + 1);
    delta_k_pla = (A_R - A_L) / (A_L + A_C + A_R);
    f_corr_pla = (idx_pla - 1 + delta_k_pla) * (f_s / N);
else
    f_corr_pla = f_axis_plot(idx_pla);
end

% =========================================================================
% 7.3 电子密度反演与误差计算
% =========================================================================
delta_f = f_corr_pla - f_corr_air;
delta_tau = delta_f / K;

% 理论近似公式参数
f_center_radar = (f_start + f_end) / 2;
const_term = (8 * pi^2 * epsilon_0 * m_e * c) / (e^2);

% 计算电子密度
n_e_trad = const_term * (f_center_radar^2 / d) * delta_tau;
err_trad = abs(n_e_trad - n_e)/n_e * 100;

% 命令行打印
fprintf('截止频率：%.0fGHz\n', f_c/1e9);
fprintf('电子密度 n_e: %.2e m^-3\n', n_e);
fprintf('校正后空气差频: %.4f MHz\n', f_corr_air/1e6);
fprintf('校正后等离子体差频: %.4f MHz\n', f_corr_pla/1e6);
fprintf('频率差: %.4f MHz\n', delta_f/1e6);
fprintf('真实电子密度: %.4e m^-3\n', n_e);
fprintf('计算电子密度: %.4e m^-3\n', n_e_trad);
fprintf('诊断误差: %.2f%%\n', err_trad);

% =========================================================================
% 7.4 绘制 Figure 10 (分析面板)
% =========================================================================
figure(10); clf;
set(gcf, 'Color', 'w', 'Position', [150, 150, 900, 750]);

% 统一显示视野 (0.6 MHz)
zoom_half_span = 0.3e6; 

% --- 子图 1: 空气 (使用加窗后的高精度数据) ---
subplot(2, 1, 1);
stem(f_axis_plot/1e6, mag_air_analysis, 'b', 'MarkerSize', 4, 'LineWidth', 1.2, 'BaseValue', 0); 
hold on;
% 标注校正线
xline(f_corr_air/1e6, 'b--', 'LineWidth', 1.5);
text(f_corr_air/1e6, val_peak_air*0.8, sprintf(' Ref Freq: %.4f MHz', f_corr_air/1e6), ...
    'Color', 'b', 'FontWeight', 'bold', 'FontSize', 10);

xlim([(f_corr_air - zoom_half_span)/1e6, (f_corr_air + zoom_half_span)/1e6]);
ylim([0, val_peak_air * 1.15]);
title('[分析] 空气介质频谱 (加窗优化以提高基准精度)'); ylabel('幅度'); grid on;

% --- 子图 2: 等离子体 (数据源同 Fig 8) ---
subplot(2, 1, 2);
stem(f_axis_plot/1e6, mag_pla_analysis, 'r', 'MarkerSize', 4, 'LineWidth', 1.2, 'BaseValue', 0); 
hold on;
% 标注校正线
xline(f_corr_pla/1e6, 'r--', 'LineWidth', 1.5);
text(f_corr_pla/1e6, val_peak_pla*0.8, sprintf(' Test Freq: %.4f MHz', f_corr_pla/1e6), ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 10);

xlim([(f_corr_pla - zoom_half_span)/1e6, (f_corr_pla + zoom_half_span)/1e6]);
ylim([0, val_peak_pla * 1.15]);
xlabel('频率 (MHz)'); ylabel('幅度'); 
title('[分析] 等离子体频谱 (色散展宽)'); grid on;

% --- 结果标注框 ---
annotation('textbox', [0.15, 0.48, 0.7, 0.05], ...
    'String', ['频差 \Deltaf = ' num2str(delta_f/1e6, '%.4f') ' MHz  =>  诊断误差 Error = ' num2str(err_trad, '%.2f') '%'], ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'k', 'LineWidth', 1, ...
    'FontWeight', 'bold', 'FontSize', 12, 'BackgroundColor', [1 1 0.9]);