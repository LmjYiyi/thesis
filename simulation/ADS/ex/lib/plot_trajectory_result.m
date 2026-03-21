function plot_trajectory_result(result, cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 统一轨迹结果绘图与导出
% 根据 cfg.enable_adaptive 和 cfg.cfg_refine 自动选择绘图模式
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pp = trajectory_postprocess();
lib_dir = fileparts(mfilename('fullpath'));
script_dir = fileparts(lib_dir);   % ex/
final = result.final;

% S2P 原始群时延曲线（与 plot_s2p_group_delay_curve.m 同源）
s2p_file = fullfile(script_dir, 'data', 'HXLBQ-DTA1329-1-1.s2p');
[f_s2p_hz, tau_s2p_s] = read_s21_group_delay_from_s2p(s2p_file);
has_s2p_curve = ~isempty(f_s2p_hz) && ~isempty(tau_s2p_s);

if has_s2p_curve
    mask_view = f_s2p_hz >= 36.5e9 & f_s2p_hz <= 37.5e9;
    if any(mask_view)
        tau_view_ns = tau_s2p_s(mask_view) * 1e9;
        fprintf('\n[s2p curve] 36.5-37.5 GHz, tau_mid=%.2f ns, tau_peak=%.2f ns\n', ...
            median(tau_view_ns), max(tau_view_ns));
    end
else
    warning('plot_trajectory_result: cannot read s2p curve: %s', s2p_file);
end

figure('Color', 'w', 'Position', [100, 100, 960, 540]);
hold on;

if cfg.enable_adaptive
    % 基础散点（灰色背景）
    scatter(result.base_cal.f_probe / 1e9, result.base_cal.tau * 1e9, 28, ...
        [0.80 0.80 0.80], 'filled', ...
        'MarkerFaceAlpha', 0.22, 'MarkerEdgeColor', 'none');

    mask_base = final.source_code == 1;
    mask_adapt = final.source_code == 2;
    mask_dense = final.source_code == 3;

    h_base = scatter(final.f_probe(mask_base) / 1e9, final.tau(mask_base) * 1e9, ...
        42, [0.16 0.46 0.72], 'filled', 'MarkerFaceAlpha', 0.90, ...
        'MarkerEdgeColor', [0.08 0.08 0.08], 'LineWidth', 0.4);

    h_adapt = scatter(final.f_probe(mask_adapt) / 1e9, final.tau(mask_adapt) * 1e9, ...
        50, [0.90 0.50 0.12], 'filled', 'MarkerFaceAlpha', 0.94, ...
        'MarkerEdgeColor', [0.08 0.08 0.08], 'LineWidth', 0.4);

    legend_handles = [h_base, h_adapt];
    legend_labels = {'固定窗口散点', '自适应窗口散点'};

    if any(mask_dense)
        h_dense = scatter(final.f_probe(mask_dense) / 1e9, final.tau(mask_dense) * 1e9, ...
            58, [0.18 0.62 0.38], 'filled', 'MarkerFaceAlpha', 0.96, ...
            'MarkerEdgeColor', [0.08 0.08 0.08], 'LineWidth', 0.4);
        switch cfg.cfg_refine.mode
            case 'mirror'
                rebuild_label = '右边缘镜像重建';
            case 'data_driven'
                rebuild_label = '右侧局部连续性重建';
            otherwise
                rebuild_label = '边缘重建';
        end


        legend_handles(end + 1) = h_dense;
        legend_labels{end + 1} = rebuild_label;
    end

    if has_s2p_curve
        h_s2p = plot(f_s2p_hz / 1e9, tau_s2p_s * 1e9, '-', ...
            'Color', [0.78 0.12 0.12], 'LineWidth', 1.9);
        legend_handles(end + 1) = h_s2p;
        legend_labels{end + 1} = 'S2P群时延曲线';
    end

    xline(cfg.cfg_hybrid.f_flat_lo / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
    xline(cfg.cfg_hybrid.f_flat_hi / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
else
    h_simple = scatter(final.f_probe / 1e9, final.tau * 1e9, 40, ...
        [0.55 0.55 0.55], 'filled', ...
        'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');

    legend_handles = h_simple;
    legend_labels = {'ESPRIT散点'};

    if has_s2p_curve
        h_s2p = plot(f_s2p_hz / 1e9, tau_s2p_s * 1e9, '-', ...
            'Color', [0.78 0.12 0.12], 'LineWidth', 1.9);
        legend_handles(end + 1) = h_s2p;
        legend_labels{end + 1} = 'S2P群时延曲线';
    end
end

hold off;

grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title(cfg.title_str, 'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);

if isfield(cfg, 'xlim_range') && numel(cfg.xlim_range) == 2
    xlim(cfg.xlim_range);
else
    xlim([36.5, 37.5]);
end

legend(legend_handles, legend_labels, 'Location', 'northeast', 'FontSize', 11);

try
    pp.export_figure(gcf, cfg.export_name, 14, 300);
catch ME
    warning('plot_trajectory_result: export failed (%s)', ME.message);
end
pp.print_diagnostics(final, cfg.export_name, cfg);
print_s2p_deviation_report(final, cfg);

mask_right = final.f_probe > cfg.cfg_hybrid.f_flat_hi;
if any(mask_right)
    fprintf('\n  -- 右侧 (>%.2f GHz) --\n', cfg.cfg_hybrid.f_flat_hi / 1e9);
    fprintf('  点数: %d, 时延: %.2f - %.2f ns\n', ...
        sum(mask_right), min(final.tau(mask_right)) * 1e9, max(final.tau(mask_right)) * 1e9);
end

fprintf('\n频率轴绘图已完成。\n');
end

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
        f_hz(end + 1, 1) = row(1) * unit_scale; %#ok<AGROW>
        s21_a(end + 1, 1) = row(4); %#ok<AGROW>
        s21_b(end + 1, 1) = row(5); %#ok<AGROW>
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
