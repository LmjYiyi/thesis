%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 图3-4：不同色散强度下差频信号的FFT频谱特征
% 功能：展示不同截止频率(等离子体频率)对应的频谱散焦效应
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% 全局字体设置（解决中文显示，带回退）
font_cn = 'SimSun';
try
    listfonts(font_cn);
catch
    font_cn = 'Microsoft YaHei';
end
set(groot, 'defaultTextFontName', font_cn);
set(groot, 'defaultAxesFontName', font_cn);

%% 1. 仿真参数设置
% LFMCW雷达参数
f_start = 34.2e9;            
f_end = 38e9;              
T_m = 50e-6;                 
B = f_end - f_start;         
K = B/T_m;                   
f_s = 80e9;                  

% 传播介质参数 (固定)
tau_air = 4e-9;              
tau_fs = 1.75e-9;            
d = 1110e-3;                  
nu = 1.5e9;                  

% 物理常量
c = 3e8;                     
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e = 1.602e-19;

% 时间轴与频率轴
t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% FFT频率轴 (包含负频率)
f = (0:N-1)*(f_s/N);         
idx_neg = f >= f_s/2;
f(idx_neg) = f(idx_neg) - f_s;
omega = 2*pi*f;              

%% 2. 截止频率数组 (对应不同色散强度)
f_c_array = [25e9, 31e9, 34.1e9];  % 弱色散、中等色散、强色散
n_cases = length(f_c_array);

% 预分配存储
S_IF_plasma_all = zeros(n_cases, N);
bandwidth_3dB = zeros(1, n_cases);

%% 3. LFMCW信号生成
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              

%% 4. 循环处理不同截止频率
for idx = 1:n_cases
    f_c = f_c_array(idx);
    omega_p = 2*pi*f_c;
    
    % 计算电子密度 (仅用于记录)
    n_e = (omega_p^2 * epsilon_0 * m_e) / e^2;
    fprintf('处理 f_p = %.0f GHz, n_e = %.2e m^-3\n', f_c/1e9, n_e);
    
    %% 4.1 信号传播模拟
    % 第一段：自由空间延迟
    delay_samples_fs = round(tau_fs/t_s);
    s_after_fs1 = [zeros(1, delay_samples_fs) s_tx(1:end-delay_samples_fs)];
    
    % 第二段：等离子体传播 (频域处理)
    S_after_fs1 = fft(s_after_fs1);
    
    % 防止除以零
    omega_safe = omega; 
    omega_safe(omega_safe == 0) = 1e-10; 
    
    % Drude模型复介电常数
    epsilon_r_complex = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
    epsilon_r_complex(omega == 0) = 1; 
    
    % 复波数
    k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
    
    % 传递函数 (强制衰减)
    k_real = real(k_complex);
    k_imag = imag(k_complex);
    H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);
    
    % 应用传递函数
    S_after_plasma = S_after_fs1 .* H_plasma;
    s_after_plasma = real(ifft(S_after_plasma));
    
    % 第三段：自由空间
    s_rx_plasma = [zeros(1, delay_samples_fs) s_after_plasma(1:end-delay_samples_fs)];
    
    %% 4.2 混频与低通滤波
    s_mix_plasma = s_tx .* real(s_rx_plasma);
    
    % 低通滤波器
    fc_lp = 100e6;
    [b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
    s_if_plasma = filtfilt(b_lp, a_lp, s_mix_plasma);
    
    %% 4.3 FFT频谱分析 (加汉宁窗)
    win = hann(N)';
    s_if_plasma_win = s_if_plasma .* win;
    S_IF_plasma = fft(s_if_plasma_win, N);
    S_IF_plasma_mag = abs(S_IF_plasma) * 2;  % 补偿窗函数幅度损失
    
    % 存储结果
    S_IF_plasma_all(idx, :) = S_IF_plasma_mag;
    
    %% 4.4 计算3dB带宽
    % 限制频率范围
    f_if_max = 2e6;  % 2 MHz
    idx_range = find(f >= 0 & f <= f_if_max);
    
    mag_range = S_IF_plasma_mag(idx_range);
    f_range = f(idx_range);
    
    % 找峰值
    [peak_val, peak_idx] = max(mag_range);
    threshold_3dB = peak_val / sqrt(2);  % -3dB
    
    % 找3dB带宽 (简单方法)
    above_threshold = mag_range >= threshold_3dB;
    first_idx = find(above_threshold, 1, 'first');
    last_idx = find(above_threshold, 1, 'last');
    
    if ~isempty(first_idx) && ~isempty(last_idx)
        bandwidth_3dB(idx) = f_range(last_idx) - f_range(first_idx);
    end
    
    fprintf('  3dB带宽: %.1f KHz\n', bandwidth_3dB(idx)/1e3);
end

%% 5. 可视化 - 横向排列子图
figure('Position', [100, 200, 1300, 360], 'Color', 'w');
tiledlayout(1, n_cases, 'TileSpacing', 'compact', 'Padding', 'compact');

% 频率显示范围
f_display_max = 1.5e6;  % 1.5 MHz
idx_display = find(f >= 0 & f <= f_display_max);

% 标题标签（与截止频率数组一致）
label_names = {'(a) 弱色散', '(b) 中等色散', '(c) 强色散'};
titles = cell(1, n_cases);
for i = 1:n_cases
    if i <= numel(label_names)
        titles{i} = sprintf('%s f_p = %.0f GHz', label_names{i}, f_c_array(i)/1e9);
    else
        titles{i} = sprintf('(case %d) f_p = %.0f GHz', i, f_c_array(i)/1e9);
    end
end
colors = [0.2 0.4 0.8; 0.2 0.6 0.4; 0.8 0.3 0.2];

for idx = 1:n_cases
    nexttile;
    
    % 归一化幅度
    mag_plot = S_IF_plasma_all(idx, idx_display);
    mag_normalized = mag_plot / max(mag_plot);
    
    % 绘制离散谱线（与图2风格一致）
    stem(f(idx_display)/1e6, mag_normalized, 'Color', colors(idx,:), ...
        'LineWidth', 1.1, 'Marker', 'none');
    hold on;
    
    % 标注3dB带宽
    yline(1/sqrt(2), 'r--', 'LineWidth', 1);
    
    % 添加带宽标注
    text(0.95, 0.85, sprintf('\\Delta f_{3dB} \\approx %.0f KHz', bandwidth_3dB(idx)/1e3), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'FontSize', 11, 'FontName', 'Times New Roman');
    
    xlabel('频率 (MHz)', 'FontSize', 12, 'FontName', font_cn);
    if idx == 1
        ylabel('归一化幅度', 'FontSize', 12, 'FontName', font_cn);
    end
    title(titles{idx}, 'FontSize', 12, 'FontName', font_cn);
    
    xlim([0, f_display_max/1e6]);
    ylim([0, 1.1]);
    grid on;
    set(gca, 'FontSize', 11, 'FontName', font_cn);
end

%% 6. 保存图像
output_dir = fileparts(mfilename('fullpath'));
fig_dir = fullfile(output_dir, '..', 'figures');

% 保存为多种格式
export_thesis_figure(gcf, '图3-4_频谱散焦效应对比', 14, 300, 'SimHei');

fprintf('\n图像已保存至: %s\n', fig_dir);
fprintf('完成！\n');
