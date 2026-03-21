%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 单独绘制 S2P 群时延曲线
% 用途：仅读取 data/HXLBQ-DTA1329-1-1.s2p 并显示 S21 群时延原始数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. 参数设置
script_dir = fileparts(mfilename('fullpath'));
s2p_file = fullfile(script_dir, 'data', 'HXLBQ-DTA1329-1-1.s2p');
x_plot_lo = 36.5e9;
x_plot_hi = 37.5e9;

fprintf('【步骤1】读取 S2P 文件并计算群时延...\n');
[f_hz, tau_s] = read_s21_group_delay_from_s2p(s2p_file);

if isempty(f_hz) || isempty(tau_s)
    error('S2P 数据为空或读取失败：%s', s2p_file);
end

%% 2. 仅绘制 S2P 原始数据
fprintf('【步骤2】绘制 S2P 原始群时延曲线...\n');
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100, 100, 980, 560]);

plot(f_hz / 1e9, tau_s * 1e9, '-', ...
    'Color', [0.12, 0.36, 0.78], 'LineWidth', 1.4);

grid on;
xlabel('频率 (GHz)', 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau_g (ns)', 'FontName', 'SimHei', 'FontSize', 12, 'FontWeight', 'bold');
title('HXLBQ-DTA1329-1-1 S21 群时延（S2P数据）', 'FontName', 'SimHei', 'FontSize', 14);
legend({'S2P 群时延数据'}, 'Location', 'best', 'FontName', 'SimHei');
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);
xlim([x_plot_lo, x_plot_hi] / 1e9);

%% 3. 结果输出（36-38 GHz 视窗）
mask_view = f_hz >= x_plot_lo & f_hz <= x_plot_hi;
if ~any(mask_view)
    error('36-38 GHz 范围内没有可用数据点。');
end

f_view = f_hz(mask_view);
tau_view_ns = tau_s(mask_view) * 1e9;
[tau_peak_ns, idx_peak] = max(tau_view_ns);
tau_min_ns = min(tau_view_ns);
tau_mid_ns = median(tau_view_ns);

fprintf('\n===== 群时延统计（36.00-38.00 GHz）=====\n');
fprintf('点数          : %d\n', numel(f_view));
fprintf('最小群时延    : %.4f ns\n', tau_min_ns);
fprintf('中位群时延    : %.4f ns\n', tau_mid_ns);
fprintf('峰值群时延    : %.4f ns @ %.4f GHz\n', tau_peak_ns, f_view(idx_peak) / 1e9);
fprintf('===========================================\n');

function [f_hz, tau_s] = read_s21_group_delay_from_s2p(s2p_file)
fid = fopen(s2p_file, 'r');
if fid < 0
    f_hz = [];
    tau_s = [];
    return;
end

cleaner = onCleanup(@() fclose(fid));

unit_scale = 1;
data_fmt = 'DB';
buf = [];
f_hz = [];
s21_a = [];
s21_b = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));
    if ~ischar(line) || isempty(line)
        continue;
    end

    if startsWith(line, '!')
        continue;
    end

    if startsWith(line, '#')
        line_upper = upper([' ' line ' ']);
        if contains(line_upper, ' GHZ ')
            unit_scale = 1e9;
        elseif contains(line_upper, ' MHZ ')
            unit_scale = 1e6;
        elseif contains(line_upper, ' KHZ ')
            unit_scale = 1e3;
        else
            unit_scale = 1;
        end

        if contains(line_upper, ' RI ')
            data_fmt = 'RI';
        elseif contains(line_upper, ' MA ')
            data_fmt = 'MA';
        else
            data_fmt = 'DB';
        end
        continue;
    end

    nums = sscanf(line, '%f').';
    if isempty(nums)
        continue;
    end

    buf = [buf, nums]; %#ok<AGROW>
    while numel(buf) >= 9
        row = buf(1:9);
        buf = buf(10:end);
        f_hz(end+1, 1) = row(1) * unit_scale; %#ok<AGROW>
        s21_a(end+1, 1) = row(4); %#ok<AGROW>
        s21_b(end+1, 1) = row(5); %#ok<AGROW>
    end
end

if isempty(f_hz)
    tau_s = [];
    return;
end

switch upper(data_fmt)
    case 'RI'
        s21 = s21_a + 1i * s21_b;
    case 'MA'
        s21 = s21_a .* exp(1i * deg2rad(s21_b));
    otherwise
        s21 = 10 .^ (s21_a / 20) .* exp(1i * deg2rad(s21_b));
end

phase_rad = unwrap(angle(s21));
tau_s = -gradient(phase_rad, f_hz) / (2 * pi);

mask_ok = isfinite(f_hz) & isfinite(tau_s);
f_hz = f_hz(mask_ok);
tau_s = tau_s(mask_ok);
end
