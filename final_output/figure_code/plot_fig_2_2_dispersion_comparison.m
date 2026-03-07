%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LFMCW 扫频与差频信号示意图 (防重叠终极排版版)
% 优化内容：
% 1. [解决重叠] 扩大了 axis 绘图边界，防止文字与坐标轴外框碰撞。
% 2. [适配尺寸] 全局下调了字体大小 (适配 16cm x 10cm 的导出尺寸)。
% 3. [坐标微调] 重新调整了所有文本标签的相对位置，确保四周留白充足。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% ========================= 1. 参数设置 =========================
fi   = 0;      
B    = 0.6;    
Tm   = 0.65;   
tau  = 0.2;    % 低频端（起始点）的最大时延

K_tx = B / Tm; % 发射信号调频率

% 坐标轴与排版参数
xmax = 1.1;
ymax_bot = 0.35;        
ymax_top = 0.85;        % 略微增高顶部坐标轴，给上方文字留空间
y_shift  = 0.70;        % 增大上下两图的间距

x_tau = tau;
x_Tm  = Tm;

% 全局字体大小 (匹配 16x10 cm 的物理尺寸)
fs_math  = 13; % 公式字母大小
fs_text  = 11; % 中文标注大小
fs_title = 12; % 底部标题大小
fs_tick  = 12; % 坐标轴刻度(Tm, tau)大小

%% ========================= 2. 数据生成 =========================
% --- 发射信号 (TX) ---
t_tx = linspace(0, Tm, 300);
f_tx = fi + K_tx * t_tx;

% --- (a) 非色散条件 (RX ND) ---
t_rx_nd = t_tx + tau;
f_rx_nd = f_tx; 

fD_nd_const = K_tx * tau;
t_bd1 = linspace(0, tau, 120);
f_bd1 = K_tx * t_bd1; 
t_bd2 = linspace(tau, Tm, 180);
f_bd2 = fD_nd_const * ones(size(t_bd2)); 

% --- (b) 强色散条件 (RX DP) ---
delay_curve = tau * (1 - 0.45 * (t_tx / Tm).^2); 
t_rx_dp = t_tx + delay_curve; 
f_rx_dp = f_tx; 

t_bd3 = linspace(0, tau, 120);
f_bd3 = K_tx * t_bd3; 

t_bd4 = linspace(tau, Tm, 200); 
alpha = -0.25; 
fD_start = K_tx * tau; 
f_bd4 = fD_start + alpha * (t_bd4 - tau); 

%% ========================= 3. 颜色与画笔 =========================
color_tx   = 'k';
color_rx   = 'r';
color_grid = [0.4 0.4 0.4]; 
font_cn    = 'SimSun';

lw_main = 1.5;
lw_dash = 1.0;
lw_axis = 1.2;
lw_beat = 2.2;  

%% ========================= 4. 图窗与布局 =========================
fig = figure('Color', 'w', 'Position', [100, 80, 1400, 900]);
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% ========================= 5. 左侧：非色散 =========================
ax1 = nexttile(1); hold(ax1, 'on'); 
% 【核心修改】大幅扩大纵向和横向的显示边界，防止文字顶到边缘
axis(ax1, [-0.2, xmax+0.15, -0.25, y_shift + ymax_top + 0.25]); 
axis(ax1, 'off');

draw_standard_axes(ax1, 0, xmax, ymax_bot, '$f_D$', lw_axis, fs_math);
draw_standard_axes(ax1, y_shift, xmax, ymax_top, '$f(t)$', lw_axis, fs_math);

plot(ax1, t_bd1, f_bd1, '-', 'Color', color_rx, 'LineWidth', lw_beat);
plot(ax1, t_bd2, f_bd2, '-', 'Color', color_rx, 'LineWidth', lw_beat);
plot(ax1, t_tx, y_shift + f_tx, '-', 'Color', color_tx, 'LineWidth', lw_main);
plot(ax1, t_rx_nd, y_shift + f_rx_nd, '--', 'Color', color_rx, 'LineWidth', lw_main);

plot(ax1, [x_tau x_tau], [0, y_shift + B], '-.', 'Color', color_grid, 'LineWidth', lw_dash);
plot(ax1, [x_Tm x_Tm],   [0, y_shift + B], '-.', 'Color', color_grid, 'LineWidth', lw_dash);
plot(ax1, [x_Tm+tau x_Tm+tau], [y_shift, y_shift + B], '-.', 'Color', color_rx, 'LineWidth', lw_dash); 
plot(ax1, [0, xmax*0.9], [y_shift+B, y_shift+B], '--', 'Color', color_grid, 'LineWidth', lw_dash); 
plot(ax1, [0, x_Tm], [fD_nd_const, fD_nd_const], '--', 'Color', color_grid, 'LineWidth', lw_dash); 

text(ax1, x_tau, -0.06, '$\tau$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);
text(ax1, x_Tm,  -0.06, '$T_m$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);
text(ax1, x_tau, y_shift - 0.06, '$\tau$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);
text(ax1, x_Tm,  y_shift - 0.06, '$T_m$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);

draw_double_arrow_vert(ax1, -0.08, y_shift, y_shift+B, 0.015, lw_axis);
text(ax1, -0.13, y_shift + B/2, '$B$', 'Interpreter', 'latex', 'FontSize', fs_math, 'HorizontalAlignment', 'center');
draw_double_arrow_horiz(ax1, x_Tm, x_Tm+tau, y_shift + 0.15, 0.015, lw_axis);
text(ax1, x_Tm + tau/2, y_shift + 0.15 - 0.08, '$\tau$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center');

% 【核心修改】将文本上移并向中间靠拢，彻底避开坐标轴
text(ax1, 0.25, y_shift + B + 0.15, '发射信号频率', 'FontName', font_cn, 'FontSize', fs_text, 'HorizontalAlignment', 'center');
draw_arrow_line(ax1, 0.25, y_shift + B + 0.11, 0.35, y_shift + K_tx*0.35, lw_axis);

text_rx_x1 = 0.85; text_rx_y1 = y_shift + B + 0.15;
text(ax1, text_rx_x1, text_rx_y1, '接收信号频率', 'FontName', font_cn, 'FontSize', fs_text, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 1);
target_x1 = 0.75; target_y1 = y_shift + interp1(t_rx_nd, f_rx_nd, target_x1);
draw_arrow_line(ax1, text_rx_x1, text_rx_y1 - 0.04, target_x1, target_y1, lw_axis);

% 【核心修改】标题下移
text(ax1, xmax/2, -0.20, '(a) 非色散', 'FontName', font_cn, 'FontSize', fs_title, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

%% ========================= 6. 右侧：色散 =========================
ax2 = nexttile(2); hold(ax2, 'on'); 
% 同步扩大边界
axis(ax2, [-0.2, xmax+0.15, -0.25, y_shift + ymax_top + 0.25]); 
axis(ax2, 'off');

draw_standard_axes(ax2, 0, xmax, ymax_bot, '$f_D$', lw_axis, fs_math);
draw_standard_axes(ax2, y_shift, xmax, ymax_top, '$f(t)$', lw_axis, fs_math);

plot(ax2, t_bd3, f_bd3, '-', 'Color', color_rx, 'LineWidth', lw_beat);
plot(ax2, t_bd4, f_bd4, '-', 'Color', color_rx, 'LineWidth', lw_beat);
plot(ax2, t_tx, y_shift + f_tx, '-', 'Color', color_tx, 'LineWidth', lw_main);
plot(ax2, t_rx_dp, y_shift + f_rx_dp, '--', 'Color', color_rx, 'LineWidth', lw_main);

plot(ax2, [x_Tm x_Tm],   [0, y_shift + B], '-.', 'Color', color_grid, 'LineWidth', lw_dash);
plot(ax2, [0, xmax*0.9], [y_shift+B, y_shift+B], '--', 'Color', color_grid, 'LineWidth', lw_dash); 

text(ax2, x_Tm,  -0.06, '$T_m$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);
text(ax2, x_Tm,  y_shift - 0.06, '$T_m$', 'Interpreter', 'latex', 'FontSize', fs_tick, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 0.5);

% 同步修改文本和箭头位置
text(ax2, 0.25, y_shift + B + 0.15, '发射信号频率', 'FontName', font_cn, 'FontSize', fs_text, 'HorizontalAlignment', 'center');
draw_arrow_line(ax2, 0.25, y_shift + B + 0.11, 0.35, y_shift + K_tx*0.35, lw_axis);

text_rx_x2 = 0.85; text_rx_y2 = y_shift + B + 0.15;
text(ax2, text_rx_x2, text_rx_y2, '接收信号频率', 'FontName', font_cn, 'FontSize', fs_text, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 1);
target_x2 = 0.75; target_y2 = y_shift + interp1(t_rx_dp, f_rx_dp, target_x2);
draw_arrow_line(ax2, text_rx_x2, text_rx_y2 - 0.04, target_x2, target_y2, lw_axis);

% 标题下移
text(ax2, xmax/2, -0.20, '(b) 色散', 'FontName', font_cn, 'FontSize', fs_title, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

%% ========================= 导出图像 =========================
fig_name = 'fig_2_2_dispersion_comparison_layout_fixed';
output_dir = fullfile(fileparts(mfilename('fullpath')), 'figures_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 保持物理尺寸不变，字体已在上方全局变量中适配
fig.Units = 'centimeters';
fig.Position = [2, 2, 16, 10];

file_tiff = fullfile(output_dir, [fig_name '.tiff']);
exportgraphics(fig, file_tiff, 'Resolution', 600, 'BackgroundColor', 'white');

file_emf = fullfile(output_dir, [fig_name '.emf']);
try
    exportgraphics(fig, file_emf, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch
    print(fig, file_emf, '-dmeta');
end

fprintf('防重叠版本 TIFF: %s\n', file_tiff);
fprintf('防重叠版本 EMF: %s\n', file_emf);

%% ========================= 绘图辅助函数 =========================
% 加入 fs_math 参数，便于控制坐标轴标签字体
function draw_standard_axes(ax, base_y, x_max, y_max_rel, y_label, lw, fs_math)
    quiver(ax, 0, base_y, x_max, 0, 0, 'Color', 'k', 'LineWidth', lw, 'MaxHeadSize', 0.05, 'Clipping', 'off');
    quiver(ax, 0, base_y, 0, y_max_rel, 0, 'Color', 'k', 'LineWidth', lw, 'MaxHeadSize', 0.05, 'Clipping', 'off');
    % 微调原点0的位置，远离坐标轴
    text(ax, -0.07, base_y - 0.05, '$0$', 'Interpreter', 'latex', 'FontSize', fs_math, 'HorizontalAlignment', 'right');
    text(ax, x_max+0.05, base_y, '$t$', 'Interpreter', 'latex', 'FontSize', fs_math+2, 'VerticalAlignment', 'middle');
    % 将纵轴标签往上推，防重叠
    text(ax, 0, base_y + y_max_rel + 0.12, y_label, 'Interpreter', 'latex', 'FontSize', fs_math+2, 'HorizontalAlignment', 'center');
end

function draw_arrow_line(ax, x1, y1, x2, y2, lw)
    quiver(ax, x1, y1, x2-x1, y2-y1, 0, 'Color', 'k', 'LineWidth', lw, 'MaxHeadSize', 0.8, 'Clipping', 'off');
end

function draw_double_arrow_vert(ax, x, y1, y2, head, lw)
    plot(ax, [x x], [y1 y2], 'k-', 'LineWidth', lw, 'Clipping', 'off');
    plot(ax, [x-head, x, x+head], [y2-head*2.5, y2, y2-head*2.5], 'k-', 'LineWidth', lw, 'Clipping', 'off');
    plot(ax, [x-head, x, x+head], [y1+head*2.5, y1, y1+head*2.5], 'k-', 'LineWidth', lw, 'Clipping', 'off');
end

function draw_double_arrow_horiz(ax, x1, x2, y, head, lw)
    plot(ax, [x1 x2], [y y], 'k-', 'LineWidth', lw, 'Clipping', 'off');
    plot(ax, [x2-head*1.5, x2, x2-head*1.5], [y-head, y, y+head], 'k-', 'LineWidth', lw, 'Clipping', 'off');
    plot(ax, [x1+head*1.5, x1, x1+head*1.5], [y-head, y, y+head], 'k-', 'LineWidth', lw, 'Clipping', 'off');
end