function export_thesis_figure(fig_handle, out_name, width_cm, dpi, font_name)
% 统一论文插图风格并导出 tiff/emf
% 用法：
%   export_thesis_figure(gcf, 'figure_name', 14, 300, 'SimHei')

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
    set(ax_all(i_ax), ...
        'FontName', font_name, ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.20, ...
        'TickDir', 'out');
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
exportgraphics(fig_handle, file_emf, 'ContentType', 'vector');

fprintf('【导出】%s\n', file_tiff);
fprintf('【导出】%s\n', file_emf);
end
