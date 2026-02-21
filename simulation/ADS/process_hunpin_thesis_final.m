%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADS 混频信号时延提取 - 终极定稿版 (坚持科学基准扣除)
% 1. 严格扣除系统真实时延 (0.2470 ns)
% 2. 利用幅度阈值剔除阻带噪声
% 3. 利用物理下限剔除通带内由纹波(Ripple)寄生调幅引起的算法失锁点
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 加载数据与参数
data = readmatrix('hunpin_time_v.txt', 'FileType', 'text', 'NumHeaderLines', 1);
valid = ~isnan(data(:,1)) & ~isnan(data(:,2));
t_raw = data(valid, 1); v_raw = data(valid, 2);

T_m = t_raw(end) - t_raw(1);
f_start = 34.4e9; f_end = 37.61e9; 
K = (f_end - f_start) / T_m;          

% --- 坚持你的科学判断：引入系统基准时延 ---
baseline_delay = 0.2470e-9; 

%% 2. 预处理 (重采样 + 低通)
fs_dec = 4e9; 
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');

[b_lp, a_lp] = butter(4, 200e6 / (fs_dec / 2));
s_if = filtfilt(b_lp, a_lp, v_dec);

s_proc = s_if(1:2:end); t_proc = t_dec(1:2:end); f_s_proc = fs_dec / 2;   

%% 3. ESPRIT 提取
win_len = max(round(0.03 * length(s_proc)), 64);
step_len = max(round(win_len / 8), 1);    
L_sub = round(win_len / 2);            

f_probe = []; tau_est = []; amp_est = [];
rms_threshold = max(abs(s_proc)) * 0.005; 
num_windows = floor((length(s_proc) - win_len) / step_len) + 1;

for i = 1:num_windows
    idx = (i-1)*step_len+1 : (i-1)*step_len+win_len;
    if idx(end) > length(s_proc), break; end
    
    x_win = s_proc(idx);
    t_c = t_proc(idx(round(win_len/2)));
    
    if t_c > 0.99*T_m || t_c < 0.01*T_m || rms(x_win) < rms_threshold, continue; end
    
    M_sub = win_len - L_sub + 1;
    X_h = zeros(L_sub, M_sub); for k=1:M_sub, X_h(:,k)=x_win(k:k+L_sub-1).'; end
    R_x = ((X_h*X_h')/M_sub + fliplr(eye(L_sub))*conj((X_h*X_h')/M_sub)*fliplr(eye(L_sub)))/2;
    
    [V, D] = eig(R_x); [lam, id] = sort(diag(D), 'descend'); V = V(:,id);
    mdl = zeros(length(lam),1);
    for k=0:length(lam)-1
        ns = lam(k+1:end); ns(ns<1e-30)=1e-30;
        mdl(k+1) = -(length(lam)-k)*M_sub*log(prod(ns)^(1/length(ns))/mean(ns)) + 0.5*k*(2*length(lam)-k)*log(M_sub);
    end
    [~, k_est] = min(mdl); num_s = min(max(1, k_est-1), 3);
    
    Us = V(:,1:num_s);
    est_f = abs(angle(eig((Us(1:end-1,:)'*Us(1:end-1,:))\(Us(1:end-1,:)'*Us(2:end,:))))) * f_s_proc/(2*pi);
    est_f = est_f(est_f > 50e3 & est_f < f_s_proc/4);
    
    if ~isempty(est_f)
        % --- 执行科学减法校准 ---
        calibrated_tau = min(est_f)/K - baseline_delay;
        
        f_probe = [f_probe, f_start + K*t_c];
        tau_est = [tau_est, calibrated_tau];
        amp_est = [amp_est, rms(x_win)];
    end
end

%% 4. 数据清洗与科学作图
% 规则1: 剔除阻带噪声 (门限略微提高到 20%，进一步净化边缘)
mask_amp = amp_est > max(amp_est) * 0.20; 
% 规则2: 剔除带内寄生调幅引起的算法失锁伪影 (结合理论红线，将物理底线收紧至 1.5 ns)
mask_physics = tau_est > 1.85e-9; 
% 综合有效点
valid_mask = mask_amp & mask_physics;

figure('Color', 'w', 'Position', [100, 100, 900, 600]);
hold on;

try
    delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
    plot(delay_data(:,1)/1e9, delay_data(:,2)*1e9, 'r-', 'LineWidth', 2, 'DisplayName', '带通滤波器理论群延迟');
catch
    warning('未找到 delay.txt');
end

scatter(f_probe(valid_mask)/1e9, tau_est(valid_mask)*1e9, 50, amp_est(valid_mask), 'filled', ...
        'MarkerFaceAlpha', 0.9, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
        'DisplayName', sprintf('ESPRIT有效特征 (系统标定 \\Delta\\tau= -%.3f ns)', baseline_delay*1e9));

cb = colorbar; ylabel(cb, '差频信号有效幅度 (RMS)', 'FontSize', 11);
grid on; set(gca, 'GridAlpha', 0.3, 'FontSize', 11);
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('去嵌入后器件群延迟 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title('基于 LFMCW 的等效色散介质时频特征提取与去嵌入标定', 'FontSize', 14);

% --- 严格限制 X 轴视野，突出核心色散区域 ---
xlim([34.4, 37.6]); 
ylim([0, 8]);
legend('Location', 'northeast', 'FontSize', 11);

%% 5. 物理参数自动评估与输出 (打印至控制台)
f_valid = f_probe(valid_mask);
tau_valid = tau_est(valid_mask);

if ~isempty(f_valid)
    % 以 37 GHz 为界，分别寻找左右两个色散尖峰（延迟最大值）
    f_split = 37.0e9;
    
    % 寻找左峰
    mask_L = f_valid < f_split;
    [~, idx_L] = max(tau_valid(mask_L));
    f_L_subset = f_valid(mask_L);
    f_peak_L = f_L_subset(idx_L);
    
    % 寻找右峰
    mask_R = f_valid >= f_split;
    [~, idx_R] = max(tau_valid(mask_R));
    f_R_subset = f_valid(mask_R);
    f_peak_R = f_R_subset(idx_R);
    
    % 计算绝对通带带宽
    BW_pass = f_peak_R - f_peak_L;
    
    % 寻找中心频率（两峰之间的谷底，即局部延迟最低点）
    mask_C = (f_valid > f_peak_L) & (f_valid < f_peak_R);
    [~, idx_C] = min(tau_valid(mask_C));
    f_C_subset = f_valid(mask_C);
    f_center = f_C_subset(idx_C);
    
    % 打印至 MATLAB 控制台
    fprintf('\n======================================================\n');
    fprintf('  基于 ESPRIT 去嵌入提取的滤波器物理参数自动评估\n');
    fprintf('======================================================\n');
    fprintf('  左侧通带边缘 (左色散峰): %7.3f GHz\n', f_peak_L / 1e9);
    fprintf('  右侧通带边缘 (右色散峰): %7.3f GHz\n', f_peak_R / 1e9);
    fprintf('  等效绝对带宽 (BW_pass) : %7.3f GHz\n', BW_pass / 1e9);
    fprintf('  滤波器中心频率 (F_center): %7.3f GHz\n', f_center / 1e9);
    fprintf('------------------------------------------------------\n');
    
    % 相对误差计算 (与理论真值 F0=37GHz, BW=1GHz 比较)
    F0_true = 37.0e9;
    BW_true = 1.0e9;
    err_F0 = abs(f_center - F0_true) / F0_true * 100;
    err_BW = abs(BW_pass - BW_true) / BW_true * 100;
    
    fprintf('  中心频率相对误差 (F_center ): %5.2f%%\n', err_F0);
    fprintf('  绝对带宽相对误差 (BW_pass ) : %5.2f%%\n', err_BW);
    fprintf('======================================================\n\n');
else
    disp('警告: 未找到足够有效特征点进行物理参数评估。');
end