%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW 等离子体色散效应可视化 (图3-4 复现)
% 功能：生成不同截止频率(fp)下的差频信号频谱，展示频谱散焦/展宽效应
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

%% 1. 基础雷达与环境参数设置
% LFMCW雷达参数
f_start = 34.2e9;            
f_end   = 37.4e9;              
T_m     = 50e-6;                 
B       = f_end - f_start;         
K       = B/T_m;                   
f_s     = 80e9;   % 采样率 80GHz               

% 物理常数
c = 3e8;                     
epsilon_0 = 8.854e-12;
m_e = 9.109e-31;
e_charge = 1.602e-19;

% 传播介质参数 (固定部分)
d = 150e-3;      % 等离子体厚度 150mm
nu = 1.5e9;      % 碰撞频率 (保持适中，主要看色散)

% 时间轴构建
t_s = 1/f_s;                 
N = round(T_m/t_s);          
t = (0:N-1)*t_s;             

% 频率轴构建 (用于频域Drude计算)
f_axis = (0:N-1)*(f_s/N);         
idx_neg = f_axis >= f_s/2;
f_axis(idx_neg) = f_axis(idx_neg) - f_s;
omega = 2*pi*f_axis; 

% 防止除零
omega_safe = omega; 
omega_safe(omega_safe == 0) = 1e-10;

%% 2. 信号生成 (只生成一次发射信号)
f_t = f_start + K*mod(t, T_m);  
phi_t = 2*pi*cumsum(f_t)*t_s;   
s_tx = cos(phi_t);              
fprintf('发射信号生成完成...\n');

%% 3. 循环仿真核心：不同截止频率下的响应
% 定义要对比的截止频率数组 (单位 Hz)
fp_list = [15e9, 25e9, 29e9]; 
plot_titles = {'(a) Weak Dispersion (fp=15GHz)', ...
               '(b) Medium Dispersion (fp=25GHz)', ...
               '(c) Strong Dispersion (fp=29GHz)'};

% 准备绘图画布
figure('Name', '色散效应频谱对比', 'Color', 'w', 'Position', [100, 200, 1400, 400]);

for k = 1:length(fp_list)
    f_c_curr = fp_list(k);
    omega_p_curr = 2*pi*f_c_curr;
    
    fprintf('正在处理: fp = %.1f GHz...\n', f_c_curr/1e9);
    
    % --- A. Drude模型传播模拟 ---
    
    % 1. 计算复介电常数 (Drude Model)
    % epsilon = 1 - wp^2 / (w^2 + i*w*nu)
    epsilon_r_complex = 1 - (omega_p_curr^2) ./ (omega_safe.^2 + 1i * omega_safe * nu);
    epsilon_r_complex(omega == 0) = 1; 
    
    % 2. 计算复波数 k
    k_complex = (omega ./ c) .* sqrt(epsilon_r_complex);
    k_real = real(k_complex);
    k_imag = imag(k_complex);
    
    % 3. 传递函数 H (包含相位延迟和衰减)
    % 确保衰减项为负指数
    H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);
    
    % 4. 频域施加影响并转回时域
    S_TX_fft = fft(s_tx);
    S_RX_plasma_fft = S_TX_fft .* H_plasma;
    s_rx_plasma = real(ifft(S_RX_plasma_fft));
    
    % --- B. 混频与滤波 ---
    
    % 混频
    s_mix = s_tx .* s_rx_plasma;
    
    % 低通滤波 (关键调整：由于色散会导致频谱展宽，这里必须放宽截止频率)
    % 原代码 100MHz 可能截断强色散信号，此处设为 300MHz
    fc_lp = 300e6;  
    [b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
    s_if = filtfilt(b_lp, a_lp, s_mix);
    
    % --- C. 频谱分析 (FFT) ---
    
    % 加窗抑制旁瓣
    win = hann(N)';
    s_if_win = s_if .* win;
    
    S_IF = fft(s_if_win, N);
    S_IF_mag = abs(S_IF); % 归一化幅度以便观察形状
    S_IF_mag = S_IF_mag / max(S_IF_mag); % 归一化到 0-1
    
    % 转换频率轴用于绘图 (只看正半轴)
    f_plot_axis = (0:N/2-1)*(f_s/N);
    S_IF_plot = S_IF_mag(1:N/2);
    
    % --- D. 绘图 (Subplot) ---
    subplot(1, 3, k);
    
    % 绘制频谱 (限制显示范围 0 - 200 MHz 以凸显展宽)
    % 注意：差频本身很低，但由于色散，能量会散开
    plot(f_plot_axis/1e6, 20*log10(S_IF_plot), 'b', 'LineWidth', 1.5);
    
    % 图表美化
    grid on;
    title(plot_titles{k}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Frequency (MHz)');
    ylabel('Normalized Amplitude (dB)');
    xlim([0, 200]); % 聚焦观察 0-200MHz 区域
    ylim([-60, 0]); % 动态范围
    
    % 在图上标注电子密度
    n_e_calc = (omega_p_curr^2 * epsilon_0 * m_e) / e_charge^2;
    text(100, -10, sprintf('Ne \\approx %.2e m^{-3}', n_e_calc), ...
         'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');
end

fprintf('仿真完成。请查看 Figure 1。\n');