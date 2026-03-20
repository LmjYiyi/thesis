function plot_trajectory_result(result, cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 统一轨迹结果绘图与导出
% 根据 cfg.enable_adaptive 和 cfg.cfg_refine 自动选择绘图模式
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pp = trajectory_postprocess();
lib_dir    = fileparts(mfilename('fullpath'));
script_dir = fileparts(lib_dir);              % ex/ 根目录

final = result.final;

figure('Color', 'w', 'Position', [100, 100, 960, 540]);
hold on;

%% 参考曲线
likely_curve = likely_filter_delay_curve(final, script_dir);
fprintf('\n[likely filter curve] passband %.2f-%.2f GHz, tau_mid=%.2f ns, tau_peak=%.2f ns\n', ...
    min(likely_curve.f_ghz), max(likely_curve.f_ghz), ...
    likely_curve.tau_floor_ns, likely_curve.tau_peak_ns);

%% 绘制
if cfg.enable_adaptive
    % 多层散点模式（有自适应 + 可能有重建）
    scatter(result.base_cal.f_probe / 1e9, result.base_cal.tau * 1e9, 28, ...
        [0.80 0.80 0.80], 'filled', ...
        'MarkerFaceAlpha', 0.22, 'MarkerEdgeColor', 'none');

    mask_base  = final.source_code == 1;
    mask_adapt = final.source_code == 2;
    mask_dense = final.source_code == 3;

    h_base = scatter(final.f_probe(mask_base) / 1e9, ...
        final.tau(mask_base) * 1e9, 42, ...
        [0.16 0.46 0.72], 'filled', ...
        'MarkerFaceAlpha', 0.90, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
        'LineWidth', 0.4);
    h_adapt = scatter(final.f_probe(mask_adapt) / 1e9, ...
        final.tau(mask_adapt) * 1e9, 50, ...
        [0.90 0.50 0.12], 'filled', ...
        'MarkerFaceAlpha', 0.94, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
        'LineWidth', 0.4);

    legend_handles = [h_base, h_adapt];
    legend_labels  = {'边缘区固定窗口', '中段自适应窗口'};

    if any(mask_dense)
        h_dense = scatter(final.f_probe(mask_dense) / 1e9, ...
            final.tau(mask_dense) * 1e9, 58, ...
            [0.18 0.62 0.38], 'filled', ...
            'MarkerFaceAlpha', 0.96, 'MarkerEdgeColor', [0.08 0.08 0.08], ...
            'LineWidth', 0.4);

        % 重建图例标签取决于模式
        switch cfg.cfg_refine.mode
            case 'mirror',      rebuild_label = '右边缘镜像重建';
            case 'data_driven', rebuild_label = '右侧局部连续性重建';
            otherwise,          rebuild_label = '边缘重建';
        end
        legend_handles(end+1) = h_dense;
        legend_labels{end+1}  = rebuild_label;
    end

    h_likely = plot(likely_curve.f_ghz, likely_curve.tau_ns, '-', ...
        'Color', [0.78 0.12 0.12], 'LineWidth', 2.2);
    set(get(get(h_likely, 'Annotation'), 'LegendInformation'), ...
        'IconDisplayStyle', 'off');

    xline(cfg.cfg_hybrid.f_flat_lo / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
    xline(cfg.cfg_hybrid.f_flat_hi / 1e9, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);
else
    % baseline 简单模式
    scatter(final.f_probe / 1e9, final.tau * 1e9, 40, ...
        [0.55 0.55 0.55], 'filled', ...
        'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
    legend_handles = [];
    legend_labels  = {'ESPRIT 散点'};
end

hold off;

grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau (ns)', 'FontSize', 12, 'FontWeight', 'bold');
title(cfg.title_str, 'FontSize', 14);
set(gca, 'FontName', 'SimHei', 'FontSize', 11, 'GridAlpha', 0.25);
xlim(cfg.xlim_range);

if ~isempty(legend_handles)
    legend(legend_handles, legend_labels, 'Location', 'northeast', 'FontSize', 11);
else
    legend(legend_labels, 'Location', 'northeast', 'FontSize', 11);
end

pp.export_figure(gcf, cfg.export_name, 14, 300);

%% 诊断输出
pp.print_diagnostics(final, cfg.export_name, cfg);

%% 右侧统计
mask_right = final.f_probe > cfg.cfg_hybrid.f_flat_hi;
if any(mask_right)
    fprintf('\n  -- 右侧 (>%.2f GHz) --\n', cfg.cfg_hybrid.f_flat_hi/1e9);
    fprintf('  点数: %d, 时延: %.2f - %.2f ns\n', ...
        sum(mask_right), ...
        min(final.tau(mask_right))*1e9, ...
        max(final.tau(mask_right))*1e9);
end

fprintf('\n频率轴已完成工程校准。\n');

end
