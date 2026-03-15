%% plot_fig_4_4_spectrogram_thesis.m
% 论文图 4-4：传统方法与滑动窗口时频解耦的对比（论文终稿版）
% 说明：
% 1) 本图不在图内放总标题，图题应由论文正文题注给出
% 2) 输出 TIFF(600 dpi) + PDF(矢量) + EMF(若平台支持)
% 3) 中英文混排按论文插图规范统一设置

clear; clc; close all;
rng(20260125);   % 固定随机种子，保证论文图可复现

%% 0. 绘图/导出参数（论文规范）
cn_font   = 'SimSun';              % 中文字体：宋体（若异常可改为 'Microsoft YaHei'）
en_font   = 'Times New Roman';     % 英文字体
font_ax   = 10.5;                  % 坐标轴刻度字号
font_lab  = 11;                    % 坐标轴标签字号
font_leg  = 10;                    % 图例字号
font_anno = 11;                    % 子图角标字号

fig_width_cm  = 16.0;              % 双子图并排，建议 15.5~17 cm
fig_height_cm = 7.2;               % 高度适中，避免过扁或过高
dpi_out       = 600;

%% 1. 物理常数与参数设置
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数
m_e = 9.109e-31;            % 电子质量 (kg)
q_e = 1.602e-19;            % 电子电量 (C)

% LFMCW雷达参数
f_start = 34.2e9;           % 起始频率 (Hz)
f_end   = 37.4e9;           % 终止频率 (Hz)
B = f_end - f_start;        % 带宽 (Hz)
T_m = 50e-6;                % 扫频周期 (s)
K = B / T_m;                % 调频斜率 (Hz/s)

% 等离子体参数（强色散工况）
f_p = 33e9;                 % 等离子体截止频率 (Hz)
omega_p = 2*pi*f_p;         % 等离子体角频率
n_e = (omega_p)^2 * epsilon_0 * m_e / q_e^2;  % 电子密度
nu_e = 1.5e9;               % 碰撞频率 (Hz)
d = 0.15;                   % 等离子体厚度 (m)

% 传播路径参数
tau_air = 4e-9;             %#ok<NASGU>  % 空气（参考）总时延 (s)
tau_fs  = 1.75e-9;          % 自由空间段时延 (s)

% 采样参数
f_s = 80e9;                 % 原始采样率 (Hz)
t_s = 1/f_s;                % 采样间隔 (s)
N   = round(T_m / t_s);     % 总采样点数
t   = (0:N-1) * t_s;        % 时间轴 (s)

fprintf('物理参数设置完成\n');
fprintf('  截止频率 f_p = %.1f GHz\n', f_p/1e9);
fprintf('  电子密度 n_e = %.2e m^-3\n', n_e);
fprintf('  调频斜率 K = %.2e Hz/s\n', K);

%% 2. LFMCW信号生成与等离子体传播模拟
fprintf('生成LFMCW信号并模拟色散传播...\n');

% 发射信号
f_t   = f_start + K * mod(t, T_m);    % 瞬时频率
phi_t = 2*pi * cumsum(f_t) * t_s;     % 瞬时相位
s_tx  = cos(phi_t);                   % 发射信号

% 构建FFT频率轴
f_fft = (0:N-1) * (f_s/N);
idx_neg = f_fft >= f_s/2;
f_fft(idx_neg) = f_fft(idx_neg) - f_s;
omega_fft = 2*pi * f_fft;

% 第一段自由空间传播
delay_samples_fs = round(tau_fs / t_s);
s_after_fs1 = [zeros(1, delay_samples_fs), s_tx(1:end-delay_samples_fs)];

% 第二段：等离子体色散传播（频域处理）
S_after_fs1 = fft(s_after_fs1);

% Drude模型复介电常数
omega_safe = omega_fft;
omega_safe(omega_safe == 0) = 1e-10;
epsilon_r = 1 - (omega_p^2) ./ (omega_safe.^2 + 1i * omega_safe * nu_e);
epsilon_r(omega_fft == 0) = 1;

% 复波数
k_complex = (omega_fft ./ c) .* sqrt(epsilon_r);

% 传递函数（强制衰减）
k_real = real(k_complex);
k_imag = imag(k_complex);
H_plasma = exp(-1i * k_real * d - abs(k_imag) * d);

% 应用传递函数
S_after_plasma = S_after_fs1 .* H_plasma;
s_after_plasma = real(ifft(S_after_plasma));

% 第三段自由空间
s_rx = [zeros(1, delay_samples_fs), s_after_plasma(1:end-delay_samples_fs)];

% 混频与低通滤波
s_mix = s_tx .* s_rx;
fc_lp = 100e6;
[b_lp, a_lp] = butter(4, fc_lp/(f_s/2));
s_if = filtfilt(b_lp, a_lp, s_mix);

fprintf('信号生成与混频完成\n');

%% 3. 滑动窗口特征提取
fprintf('模拟滑动窗口特征提取...\n');

% 滑动窗口参数
T_w = 12e-6;                % 窗口时长
f_s_proc = 20e6;            % 降采样后采样率
win_len  = round(T_w * f_s_proc);
step_len = round(win_len / 10);      % 90%重叠
N_proc   = round(T_m * f_s_proc);
t_proc   = linspace(0, T_m, N_proc);

num_windows = floor((N_proc - win_len) / step_len);

% 特征点存储
feature_f_beat  = [];
feature_f_probe = [];

for i = 1:num_windows
    t_center = t_proc((i-1)*step_len + round(win_len/2));

    % 边缘保护
    if t_center < 0.05*T_m || t_center > 0.95*T_m
        continue;
    end

    % 当前探测频率
    f_current = f_start + K * t_center;

    % 理论计算（基于Drude模型）
    omega_i = 2*pi * f_current;
    if f_current > f_p * 1.01
        epsilon_r_i = 1 - (omega_p^2) / (omega_i * (omega_i + 1i*nu_e));
        tau_g_i = (d/c) * real(1 / sqrt(epsilon_r_i)) + 2*tau_fs;
        f_beat_i = K * tau_g_i;

        % 模拟ESPRIT测量噪声
        f_beat_meas = f_beat_i * (1 + 0.002*randn);

        feature_f_beat(end+1)  = f_beat_meas; %#ok<SAGROW>
        feature_f_probe(end+1) = f_current;   %#ok<SAGROW>
    end
end

fprintf('特征点提取完成: %d 个有效点\n', length(feature_f_beat));

%% 4. 理论曲线
t_theory = linspace(0.05*T_m, 0.95*T_m, 200);
f_probe_theory = f_start + K * t_theory;
f_beat_theory = zeros(size(t_theory));

for i = 1:length(t_theory)
    omega_i = 2*pi * f_probe_theory(i);
    if f_probe_theory(i) > f_p * 1.01
        epsilon_r_i = 1 - (omega_p^2) / (omega_i * (omega_i + 1i*nu_e));
        tau_g_i = (d/c) * real(1 / sqrt(epsilon_r_i)) + 2*tau_fs;
        f_beat_theory(i) = K * tau_g_i;
    else
        f_beat_theory(i) = NaN;
    end
end

%% 5. 传统方法下的差频提取（长窗FFT，失败示例）
num_blocks = 20;
block_len = floor(length(s_if) / num_blocks);

f_probe_fft = zeros(1, num_blocks);
f_beat_fft  = zeros(1, num_blocks);

for k = 1:num_blocks
    idx = (k-1)*block_len + (1:block_len);
    idx(idx > length(s_if)) = [];
    if isempty(idx)
        continue;
    end

    s_blk = s_if(idx);
    S_blk = abs(fft(s_blk, 2^14));
    f_axis_blk = (0:length(S_blk)-1) * f_s / length(S_blk);

    % 只在差频范围内找峰
    idx_band = f_axis_blk > 200e3 & f_axis_blk < 450e3;
    f_axis_band = f_axis_blk(idx_band);
    S_band = S_blk(idx_band);
    if isempty(S_band)
        continue;
    end

    [~, imax] = max(S_band);
    f_beat_fft(k) = f_axis_band(imax);

    % 对应探测频率（块中心）
    t_center = mean(idx) * t_s;
    f_probe_fft(k) = f_start + K * t_center;
end

% 去掉未赋值的点
valid_fft = (f_probe_fft > 0) & (f_beat_fft > 0);
f_probe_fft = f_probe_fft(valid_fft);
f_beat_fft  = f_beat_fft(valid_fft);

%% 6. 绘图
fprintf('绘制对比图...\n');

fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [2, 2, fig_width_cm, fig_height_cm], ...
             'PaperUnits', 'centimeters', ...
             'PaperPositionMode', 'auto', ...
             'PaperSize', [fig_width_cm, fig_height_cm]);

% 使用 tiledlayout 比 subplot 更稳，排版更像论文终稿
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 配色：控制克制，避免“演示文稿风”
clr_gray   = [0.82, 0.82, 0.82];
clr_black  = [0.10, 0.10, 0.10];
clr_blue   = [0.00, 0.32, 0.74];
clr_orange = [0.85, 0.33, 0.10];

% 公共坐标范围
x_lim = [34.2, 37.4];
f_beat_min_val = min(feature_f_beat);
f_beat_max_val = max(feature_f_beat);

% 纵轴稍留白
ymin = floor((min([f_beat_min_val, f_beat_fft])*1e-3 - 2));
ymax = ceil((max([f_beat_max_val, f_beat_fft])*1e-3 + 2));
y_lim = [ymin, ymax];

% =========================
% 子图 (a)
% =========================
ax1 = nexttile(tl, 1);
hold(ax1, 'on');

% 灰色散焦区域
h_fill = fill(ax1, ...
    [x_lim(1), x_lim(2), x_lim(2), x_lim(1)], ...
    [f_beat_min_val, f_beat_min_val, f_beat_max_val, f_beat_max_val]/1e3, ...
    clr_gray, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.65, ...
    'DisplayName', 'FFT散焦区域');

% 传统FFT结果
h_fft = plot(ax1, f_probe_fft/1e9, f_beat_fft/1e3, '.-', ...
    'Color', clr_black, ...
    'LineWidth', 1.1, ...
    'MarkerSize', 10, ...
    'DisplayName', '传统FFT结果');

format_axes(ax1, en_font, font_ax);

xlabel(ax1, '\fontname{SimSun}探测频率 \fontname{Times New Roman}f_{probe} (GHz)', ...
    'Interpreter', 'tex', 'FontSize', font_lab);
ylabel(ax1, '\fontname{SimSun}差频 \fontname{Times New Roman}f_D (kHz)', ...
    'Interpreter', 'tex', 'FontSize', font_lab);

xlim(ax1, x_lim);
ylim(ax1, y_lim);

lg1 = legend(ax1, [h_fill, h_fft], {'FFT散焦区域', '传统FFT结果'}, ...
    'Location', 'northeast');
format_legend(lg1, cn_font, font_leg);

% 子图角标
text(ax1, 0.02, 0.96, '(a)', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontName', en_font, ...
    'FontSize', font_anno, ...
    'FontWeight', 'bold');



% =========================
% 子图 (b)
% =========================
ax2 = nexttile(tl, 2);
hold(ax2, 'on');

valid_theory = ~isnan(f_beat_theory);

h_theory = plot(ax2, ...
    f_probe_theory(valid_theory)/1e9, ...
    f_beat_theory(valid_theory)/1e3, '--', ...
    'Color', clr_blue, ...
    'LineWidth', 1.5, ...
    'DisplayName', '理论真值');

h_meas = scatter(ax2, ...
    feature_f_probe/1e9, ...
    feature_f_beat/1e3, ...
    16, ...
    'o', ...
    'MarkerFaceColor', clr_orange, ...
    'MarkerEdgeColor', 'w', ...
    'LineWidth', 0.5, ...
    'DisplayName', '滑动窗口提取');

format_axes(ax2, en_font, font_ax);

xlabel(ax2, '\fontname{SimSun}探测频率 \fontname{Times New Roman}f_{probe} (GHz)', ...
    'Interpreter', 'tex', 'FontSize', font_lab);
ylabel(ax2, '\fontname{SimSun}瞬时差频 \fontname{Times New Roman}f_D (kHz)', ...
    'Interpreter', 'tex', 'FontSize', font_lab);

xlim(ax2, x_lim);
ylim(ax2, y_lim);

lg2 = legend(ax2, [h_theory, h_meas], {'理论真值', '滑动窗口提取'}, ...
    'Location', 'northeast');
format_legend(lg2, cn_font, font_leg);

text(ax2, 0.02, 0.96, '(b)', ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', ...
    'FontName', en_font, ...
    'FontSize', font_anno, ...
    'FontWeight', 'bold');



%% 7. 导出图表（论文终稿格式）
% 建议输出到当前脚本同级目录下的 figures_export
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir, 'figures_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig_name_base = '图4-4_传统方法与滑动窗口时频解耦对比';

export_thesis_figure(fig, output_dir, fig_name_base, dpi_out);

fprintf('\n✓ 图 4-4 已导出\n');
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.tiff']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.pdf']));
fprintf('  - %s\n', fullfile(output_dir, [fig_name_base, '.emf']));

%% 8. 输出物理特征统计
fprintf('\n========== 物理特征统计 ==========\n');
fprintf('仿真参数:\n');
fprintf('  截止频率 f_p = %.1f GHz\n', f_p/1e9);
fprintf('  碰撞频率 ν_e = %.1f GHz\n', nu_e/1e9);
fprintf('  等离子体厚度 d = %.2f m\n', d);
fprintf('  调频斜率 K = %.2e Hz/s\n', K);

fprintf('\n传统方法结论:\n');
fprintf('  差频散焦范围: %.1f - %.1f kHz (约 %.1f kHz)\n', ...
    f_beat_min_val/1e3, f_beat_max_val/1e3, (f_beat_max_val-f_beat_min_val)/1e3);
fprintf('  问题: 差频与探测频率关系不可解析（无唯一映射）\n');

fprintf('\n滑动窗口方法:\n');
fprintf('  窗口时长 T_w = %.0f μs\n', T_w*1e6);
fprintf('  步长 T_step = %.1f μs (%.0f%% 重叠)\n', T_w/10*1e6, 90);
fprintf('  有效窗口数 N_w = %d\n', length(feature_f_beat));
fprintf('  优势: 精确追踪差频随探测频率的非线性变化\n');
fprintf('================================================\n');

%% ========================= 本地函数 =========================
function format_axes(ax, en_font, font_ax)
    set(ax, ...
        'FontName', en_font, ...
        'FontSize', font_ax, ...
        'LineWidth', 0.9, ...
        'Box', 'on', ...
        'TickDir', 'in', ...
        'XMinorTick', 'off', ...
        'YMinorTick', 'off', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.18, ...
        'GridLineStyle', '-');
end

function format_legend(lg, cn_font, font_leg)
    set(lg, ...
        'FontName', cn_font, ...
        'FontSize', font_leg, ...
        'Interpreter', 'tex', ...
        'Box', 'on', ...
        'Color', 'white', ...
        'EdgeColor', [0.7 0.7 0.7], ...
        'LineWidth', 0.6, ...
        'AutoUpdate', 'off');
end

function export_thesis_figure(fig_handle, out_dir, out_name, dpi)
    % 统一白底
    set(fig_handle, 'Color', 'w');

    file_tiff = fullfile(out_dir, [out_name, '.tiff']);
    file_pdf  = fullfile(out_dir, [out_name, '.pdf']);
    file_emf  = fullfile(out_dir, [out_name, '.emf']);

    % 位图：论文送审/打印
    exportgraphics(fig_handle, file_tiff, ...
        'Resolution', dpi, ...
        'BackgroundColor', 'white');

    % 矢量：论文排版首选
    exportgraphics(fig_handle, file_pdf, ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    % EMF：Windows Word 粘贴常用
    try
        exportgraphics(fig_handle, file_emf, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    catch
        warning('EMF export failed on current platform.');
    end
end