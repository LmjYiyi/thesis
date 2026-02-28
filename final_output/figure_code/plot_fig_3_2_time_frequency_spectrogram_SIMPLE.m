%% ===============================
% 图3-2 差频信号瞬时频率演化对比（简化版）
% 章节：3.3.2 差频信号相位的非线性畸变与瞬时频率解析
% 核心对比：无色散（水平线） vs 强色散（斜线）
% ===============================

clear; clc; close all;

%% ---------- 中文字体设置 ----------
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');

%% ---------- 参数设置 ----------
Tm = 1e-3;                 % 扫频周期 1 ms
fs = 50e6;                 % 采样率 50 MHz
t = 0:1/fs:Tm-1/fs;        % 时间轴

fD0 = 5e6;                 % 差频中心频率 5 MHz

% 色散参数（控制频率漂移斜率）
alpha0 = 0;                % 无色散 α = 0
alpha1 = 1.2e10;           % 强色散 α (Hz/s) —— 可调参数，控制斜率大小

%% ---------- 构造差频信号 ----------
% (a) 无色散：单频信号，相位线性增长
s_no_disp = exp(1j*2*pi*fD0*t);

% (b) 强色散：Chirp信号，相位包含二次项
% 相位 φ(t) = 2π*fD0*t + π*α*t²
s_disp = exp(1j*(2*pi*fD0*t + pi*alpha1*t.^2));

%% ---------- 瞬时频率计算 ----------
% 瞬时相位解包裹
phi_no = unwrap(angle(s_no_disp));
phi_di = unwrap(angle(s_disp));

% 瞬时频率 f(t) = (1/2π) dφ/dt
f_inst_no = (1/(2*pi))*gradient(phi_no, 1/fs);
f_inst_di = (1/(2*pi))*gradient(phi_di, 1/fs);

%% ---------- 绘图（博士论文标准格式） ----------
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [5 5 16 6]);

% 设置纸张属性（解决PDF裁剪问题）
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperSize', [16 6]);
set(fig, 'PaperPosition', [0 0 16 6]);

% ----- 子图(a)：无色散 -----
subplot(1,2,1)
plot(t*1e3, f_inst_no/1e6, 'b', 'LineWidth', 2)
grid on; box on
set(gca, 'FontSize', 11, 'LineWidth', 1)
xlabel('时间 {\itt} (ms)', 'FontSize', 12)
ylabel('瞬时频率 {\itf}_D (MHz)', 'FontSize', 12)
title('(a) 无色散 (\alpha = 0)', 'FontSize', 12, 'FontWeight', 'normal')
ylim([fD0/1e6-1 fD0/1e6+1])
xlim([0 Tm*1e3])

% ----- 子图(b)：强色散 -----
subplot(1,2,2)
plot(t*1e3, f_inst_di/1e6, 'r', 'LineWidth', 2)
grid on; box on
set(gca, 'FontSize', 11, 'LineWidth', 1)
xlabel('时间 {\itt} (ms)', 'FontSize', 12)
ylabel('瞬时频率 {\itf}_D (MHz)', 'FontSize', 12)
title('(b) 强色散 ({\itf}_p \approx 30 GHz)', 'FontSize', 12, 'FontWeight', 'normal')
ylim([min(f_inst_di)/1e6-2 max(f_inst_di)/1e6+2])
xlim([0 Tm*1e3])

%% ---------- 图像保存 ----------
output_dir = '../figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存为高分辨率格式
export_thesis_figure(fig, 'fig_3_2_instantaneous_frequency', 14, 300, 'SimHei');

fprintf('\n图3-2 已保存至 %s\n', output_dir);

%% ---------- 控制台输出物理特征 ----------
fprintf('\n========== 差频信号物理特征 ==========\n');
fprintf('(a) 无色散：f_D = %.2f MHz (常数)\n', fD0/1e6);
fprintf('(b) 强色散：f_D 范围 = [%.2f, %.2f] MHz\n', min(f_inst_di)/1e6, max(f_inst_di)/1e6);
fprintf('    净频率漂移：Δf_D = %.2f MHz\n', (max(f_inst_di)-min(f_inst_di))/1e6);
fprintf('    漂移系数 α = %.2e Hz/s\n', alpha1);
fprintf('\n调参提示：修改第20行 alpha1 值可调整斜率大小\n');