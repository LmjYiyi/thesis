function export_thesis_figure(fig_handle, out_name, width_cm, dpi, font_name)
% 统一论文插图风格并导出 TIFF/EMF
% 用法：export_thesis_figure(gcf, '图4-1_示例', 14, 300, 'SimHei')

if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 300; end
if nargin < 5 || isempty(font_name), font_name = 'SimHei'; end

height_cm = width_cm * 0.618;
out_dir = fullfile(pwd, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPosition', [0, 0, width_cm, height_cm], ...
    'PaperSize', [width_cm, height_cm]);

ax_all = findall(fig_handle, 'Type', 'axes');
for i_ax = 1:numel(ax_all)
    ax = ax_all(i_ax);
    if isprop(ax, 'Tag') && strcmpi(ax.Tag, 'legend')
        continue;
    end

    set(ax, ...
        'FontName', font_name, ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.20, ...
        'TickDir', 'out');

    x_label = get(get(ax, 'XLabel'), 'String');
    y_label = get(get(ax, 'YLabel'), 'String');
    if isempty(x_label)
        xlabel(ax, '横坐标 (单位)', 'FontName', font_name, 'FontSize', 10);
    end
    if isempty(y_label)
        ylabel(ax, '纵坐标 (单位)', 'FontName', font_name, 'FontSize', 10);
    end

    tighten_axis_limits(ax);
end

line_all = findall(fig_handle, 'Type', 'line');
for i_ln = 1:numel(line_all)
    if strcmp(get(line_all(i_ln), 'LineStyle'), 'none')
        set(line_all(i_ln), 'LineWidth', 1.0);
    else
        set(line_all(i_ln), 'LineWidth', 1.5);
    end
end

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
file_emf = fullfile(out_dir, [out_name, '.emf']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi);
try
    exportgraphics(fig_handle, file_emf, 'ContentType', 'vector');
catch
    warning('EMF 导出失败，当前平台可能不支持 EMF。');
end

fprintf('【导出】%s\n', file_tiff);
fprintf('【导出】%s\n', file_emf);
end

function tighten_axis_limits(ax)
children = allchild(ax);
x_vals = [];
y_vals = [];
for i = 1:numel(children)
    obj = children(i);
    if isprop(obj, 'XData')
        x_data = get(obj, 'XData');
        x_vals = [x_vals; x_data(:)]; %#ok<AGROW>
    end
    if isprop(obj, 'YData')
        y_data = get(obj, 'YData');
        y_vals = [y_vals; y_data(:)]; %#ok<AGROW>
    end
end

x_vals = x_vals(isfinite(x_vals));
y_vals = y_vals(isfinite(y_vals));

if ~isempty(x_vals)
    x_min = min(x_vals);
    x_max = max(x_vals);
    if x_min < x_max
        xlim(ax, [x_min, x_max]);
    end
end

if ~isempty(y_vals)
    y_min = min(y_vals);
    y_max = max(y_vals);
    if y_min < y_max
        y_pad = 0.03 * (y_max - y_min);
        ylim(ax, [y_min - y_pad, y_max + y_pad]);
    end
end
end
