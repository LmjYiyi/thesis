function export_thesis_figure(fig_handle, out_name, width_cm, dpi, cn_font)
% Standardize thesis figure style and export TIFF/EMF.

if nargin < 1 || isempty(fig_handle), fig_handle = gcf; end
if nargin < 2 || isempty(out_name), out_name = 'figure_export'; end
if nargin < 3 || isempty(width_cm), width_cm = 14; end
if nargin < 4 || isempty(dpi), dpi = 600; end
if nargin < 5 || isempty(cn_font), cn_font = 'SimSun'; end

en_font = 'Times New Roman';
height_cm = width_cm * 0.65;

out_dir = fullfile(pwd, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(fig_handle, ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [2, 2, width_cm, height_cm], ...
    'PaperUnits', 'centimeters', ...
    'PaperPositionMode', 'auto', ...
    'PaperSize', [width_cm, height_cm]);

hide_sgtitle(fig_handle);
style_axes(fig_handle, en_font, cn_font);
style_lines(fig_handle);
style_legend(fig_handle, cn_font, en_font);

file_tiff = fullfile(out_dir, [out_name, '.tiff']);
file_emf = fullfile(out_dir, [out_name, '.emf']);
exportgraphics(fig_handle, file_tiff, 'Resolution', dpi, 'BackgroundColor', 'white');
try
    exportgraphics(fig_handle, file_emf, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch
    warning('EMF export failed on current platform.');
end

fprintf('[export] %s\n', file_tiff);
fprintf('[export] %s\n', file_emf);
end

function hide_sgtitle(fig_handle)
title_nodes = [ ...
    findall(fig_handle, 'Type', 'Text', 'Tag', 'suptitle'); ...
    findall(fig_handle, 'Type', 'Text', 'Tag', 'sgtitle') ...
];
if ~isempty(title_nodes)
    set(title_nodes, 'Visible', 'off');
end
end

function style_axes(fig_handle, en_font, cn_font)
ax_all = findall(fig_handle, 'Type', 'axes');
for i = 1:numel(ax_all)
    ax = ax_all(i);
    if isprop(ax, 'Tag') && strcmpi(ax.Tag, 'legend')
        continue;
    end

    set(ax, ...
        'FontName', en_font, ...
        'FontSize', 10, ...
        'LineWidth', 1.0, ...
        'Box', 'on', ...
        'XGrid', 'on', ...
        'YGrid', 'on', ...
        'GridAlpha', 0.25, ...
        'TickDir', 'in');

    xl = ax.XLabel;
    if ~isempty(xl) && ~isempty(xl.String)
        set(xl, 'FontName', cn_font, 'FontSize', 10, 'Interpreter', 'tex');
    end

    for j = 1:numel(ax.YAxis)
        yl = ax.YAxis(j).Label;
        if ~isempty(yl) && ~isempty(yl.String)
            set(yl, 'FontName', cn_font, 'FontSize', 10, 'Interpreter', 'tex');
        end
    end

    tl = ax.Title;
    if ~isempty(tl) && ~isempty(tl.String)
        set(tl, 'FontName', cn_font, 'FontSize', 10, 'FontWeight', 'normal');
    end
end
end

function style_lines(fig_handle)
line_all = findall(fig_handle, 'Type', 'line');
for i = 1:numel(line_all)
    if strcmp(get(line_all(i), 'LineStyle'), 'none')
        set(line_all(i), 'LineWidth', 1.0);
    else
        set(line_all(i), 'LineWidth', 1.5);
    end
end
end

function style_legend(fig_handle, cn_font, en_font)
legend_all = findall(fig_handle, 'Type', 'legend');
for i = 1:numel(legend_all)
    lg = legend_all(i);

    set(lg, ...
        'FontName', cn_font, ...
        'FontSize', 10, ...
        'Interpreter', 'tex', ...
        'TextColor', 'black', ...
        'Box', 'on', ...
        'Color', 'white', ...
        'EdgeColor', [0.7 0.7 0.7], ...
        'LineWidth', 0.6, ...
        'AutoUpdate', 'off');
end
end

