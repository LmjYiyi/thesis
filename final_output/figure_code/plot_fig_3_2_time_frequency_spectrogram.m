%% plot_fig_3_2_time_frequency_spectrogram.m
% 论文图 3-2：差频信号瞬时频率演化的时频分析对比
% 生成日期：2026-01-23
% 对应章节：3.3.2 差频信号相位的非线性畸变与瞬时频率解析
%
% 图表核心表达（基于定稿文档）：
% - 对比无色散与强色散条件下差频信号的频率演化轨迹
% - (a) 理想无色散：瞬时频率为恒定水平线，α=0
% - (b) 强色散(fp=29GHz)：瞬时频率呈斜向Chirp轨迹，斜率为α
% - 核心对比："水平线 vs 斜线"的鲜明视觉冲击

clear; clc; close all;

% 字体设置（解决中文显示问题）
font_cn = 'Microsoft YaHei';
font_en = 'Times New Roman';
set(groot, 'defaultAxesFontName', font_en);
set(groot, 'defaultTextFontName', font_en);

%% 1. 物理常数与雷达参数
c = 2.99792458e8;           % 光速 (m/s)
e = 1.60217663e-19;         % 电子电荷 (C)
me = 9.10938356e-31;        % 电子质量 (kg)
eps0 = 8.85418781e-12;      % 真空介电常数 (F/m)

% 雷达参数
f0 = 34e9;                  % 中心频率 (Hz)
B = 4e9;                    % 带宽 (Hz)
Tm = 1e-3;                  % 调制周期 (s)
K = B / Tm;                 % 调频斜率 (Hz/s)
d = 0.15;                   % 厚度 (m)
tau_0 = d / c;              % 基础时延 (s)

%% 2. 计算色散参数
fp_strong = 29e9;           % 强色散截止频率 (Hz)

omega_0 = 2*pi*f0;
x = fp_strong / f0;

% 一阶和二阶色散系数
tau_1 = (1/(2*pi)) * (-tau_0/f0) * (x^2) / ((1-x^2)^1.5);
tau_2 = (1/(2*pi)^2) * (tau_0/f0^2) * ...
        ((3*x^4)/(1-x^2)^2.5 + x^2/(1-x^2)^1.5);

% 计算α（频率漂移率）
K_prime = 2*pi*K;
alpha_strong = (omega_0*tau_2*K_prime^2)/(2*pi) + 2*(B/Tm)*tau_1*K_prime ...
               - (B/Tm)*((tau_1*K_prime)^2 + 2*tau_0*tau_2*K_prime^2/2);

fprintf('════════════════════════════════════════════════════════\n');
fprintf('图3-2 仿真参数设置\n');
fprintf('════════════════════════════════════════════════════════\n');
fprintf('  中心频率 f0 = %.1f GHz\n', f0/1e9);
fprintf('  带宽 B = %.1f GHz\n', B/1e9);
fprintf('  调制周期 Tm = %.1f ms\n', Tm*1e3);
fprintf('  等离子体厚度 d = %.0f mm\n', d*1e3);
fprintf('  基础时延 τ0 = %.3f ns\n', tau_0*1e9);
fprintf('────────────────────────────────────────────────────────\n');
fprintf('  无色散：α = 0 Hz/s (水平线)\n');
fprintf('  强色散：fp = %.1f GHz, α = %.2e Hz/s (斜线)\n', fp_strong/1e9, alpha_strong);
fprintf('  频率漂移范围：Δf = %.1f kHz (1 ms周期内)\n', abs(alpha_strong)*Tm/1e3);
fprintf('════════════════════════════════════════════════════════\n\n');

%% 3. 生成时域信号
Fs = 1e6;                   % 采样率 1 MHz (提高时频分辨率)
t = 0:1/Fs:Tm-1/Fs;         % 时间轴
N = length(t);

% 中心差频
f_beat_center = K * tau_0;

% 无色散信号（单一频率，水平线）
signal_nodispersion = exp(1j * 2*pi * f_beat_center * t);

% 强色散信号（Chirp，斜线）
signal_dispersion = exp(1j * 2*pi * (f_beat_center * t + 0.5 * alpha_strong * t.^2));

%% 4. 高质量时频分析（STFT）
% 使用Hamming窗，更平滑的频谱
window_len = 256;           % 窗口长度（更好的频率分辨率）
overlap = round(0.95 * window_len);  % 95%重叠（更平滑的时频图）
nfft = 1024;                % FFT点数

% 计算时频图
[S_nodis, F_nodis, T_nodis] = spectrogram(signal_nodispersion, ...
    hamming(window_len), overlap, nfft, Fs, 'yaxis');

[S_dis, F_dis, T_dis] = spectrogram(signal_dispersion, ...
    hamming(window_len), overlap, nfft, Fs, 'yaxis');

% 归一化到dB
S_nodis_dB = 20*log10(abs(S_nodis)/max(abs(S_nodis(:))));
S_dis_dB = 20*log10(abs(S_dis)/max(abs(S_dis(:))));

%% 5. 高质量学术绘图
% 创建高分辨率figure
fig = figure('Position', [100, 100, 1400, 550], 'Color', 'w');

% 专业配色：jet的改进版（对比度更好）
custom_colormap = jet(256);
custom_colormap = flipud(custom_colormap);  % 红色为高能量

% === 子图(a)：无色散 - 水平线 ===
subplot(1,2,1);
imagesc(T_nodis*1e3, F_nodis/1e3, S_nodis_dB);
axis xy;
caxis([-50, 0]);            % 动态范围50dB
colormap(gca, custom_colormap);

hold on;
% 理论瞬时频率线（白色粗实线，强调"水平"特征）
plot(t*1e3, ones(size(t))*f_beat_center/1e3, 'w-', 'LineWidth', 3.5);
hold off;

% 坐标轴范围（聚焦信号区域）
ylim([f_beat_center/1e3 - 100, f_beat_center/1e3 + 100]);

% 坐标轴标签（加粗，学术规范）
xlabel('时间 (ms)', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', font_cn);
ylabel('频率 (kHz)', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', font_cn);

% 子图标题（清晰标注核心特征）
title({'{\bf(a) 无色散：瞬时频率恒定}', ...
       ['{\itα} = 0，{\itf}_{\itD}({\itt}) = {\itf''}_{\itD} (水平线)']}, ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontName', font_cn, 'Interpreter', 'tex');

% 添加文本标注（白色背景框，可读性强）
text(0.5, f_beat_center/1e3 + 50, '理论: f_D(t) = f''_D', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', font_cn, ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0, 0, 0, 0.7], ...
    'EdgeColor', 'w', 'LineWidth', 1.5, 'Margin', 3);

% 坐标轴美化
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'Box', 'on', 'FontName', font_en);
grid on;
ax = gca;
ax.GridColor = [1, 1, 1];
ax.GridAlpha = 0.3;
ax.Layer = 'top';

% === 子图(b)：强色散 - 斜线Chirp ===
subplot(1,2,2);
imagesc(T_dis*1e3, F_dis/1e3, S_dis_dB);
axis xy;
caxis([-50, 0]);
colormap(gca, custom_colormap);

hold on;
% 理论瞬时频率线（白色粗实线，强调"斜线"特征）
f_inst = f_beat_center + alpha_strong * t;
plot(t*1e3, f_inst/1e3, 'w-', 'LineWidth', 3.5);

% 添加斜率指示箭头（更醒目）
t_arrow = 0.6e-3;
f_arrow = f_beat_center + alpha_strong * t_arrow;
dt_arrow = 0.25e-3;
df_arrow = alpha_strong * dt_arrow;

% 绘制大箭头
quiver(t_arrow*1e3, f_arrow/1e3, dt_arrow*1e3, df_arrow/1e3, ...
       0, 'Color', [1, 1, 0], 'LineWidth', 3, 'MaxHeadSize', 2);

hold off;

% 坐标轴范围
y_center = mean(f_inst)/1e3;
ylim([y_center - 100, y_center + 100]);

% 坐标轴标签
xlabel('时间 (ms)', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', font_cn);
ylabel('频率 (kHz)', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', font_cn);

% 子图标题
title({'{\bf(b) 强色散：瞬时频率线性漂移}', ...
       sprintf('{\\itf}_{\\itp} = %.0f GHz, {\\itα} ≠ 0 (斜线Chirp)', fp_strong/1e9)}, ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontName', font_cn, 'Interpreter', 'tex');

% 标注文本框（斜率信息）
text(0.7, f_beat_center/1e3 + 70, ...
    sprintf('斜率 α = %.1e Hz/s\n频率漂移 Δf ≈ %.0f kHz', alpha_strong, abs(alpha_strong)*Tm/1e3), ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', font_cn, ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0, 0, 0, 0.7], ...
    'EdgeColor', 'w', 'LineWidth', 1.5, 'Margin', 3);

% 理论公式标注
text(0.15, y_center - 70, 'f_D(t) = f''_D + αt', ...
    'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', ...
    'BackgroundColor', [0, 0, 0, 0.7], ...
    'EdgeColor', 'w', 'LineWidth', 1.5, 'Margin', 3);

% 坐标轴美化
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'Box', 'on', 'FontName', font_en);
grid on;
ax = gca;
ax.GridColor = [1, 1, 1];
ax.GridAlpha = 0.3;
ax.Layer = 'top';

% 总标题（加粗，专业）
sgtitle('{\bf图3-2  差频信号瞬时频率演化的时频分析对比：水平线 vs 斜线}', ...
    'FontSize', 16, 'FontWeight', 'bold', 'FontName', font_cn, 'Interpreter', 'tex');

%% 6. 保存图表
output_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保存PNG（高分辨率，论文插图）
print('-dpng', '-r300', fullfile(output_dir, '图3-2_时频分析对比.png'));

% 保存SVG（矢量图，LaTeX排版）
print('-dsvg', fullfile(output_dir, '图3-2_时频分析对比.svg'));

% 保存PDF（投稿用）
print('-dpdf', fullfile(output_dir, '图3-2_时频分析对比.pdf'));

fprintf('\n✓ 图3-2 已保存至 final_output/figures/\n');
fprintf('════════════════════════════════════════════════════════\n');
fprintf('图表特征：\n');
fprintf('  左图(a)：无色散，水平线（α=0）\n');
fprintf('  右图(b)：强色散，斜线Chirp（α≠0）\n');
fprintf('  配色：jet改进版（红色=高能量，蓝色=低能量）\n');
fprintf('  对比：清晰展示"稳态 vs 非稳态"物理过程\n');
fprintf('════════════════════════════════════════════════════════\n');
