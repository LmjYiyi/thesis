%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 3-8: Multisolution intersections in group-delay trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. Physical constants
e_charge = 1.602e-19;
me = 9.109e-31;
eps0 = 8.854e-12;
c = 3e8;

nu_val = 1.5e9;
f_probe = linspace(34e9, 37.5e9, 600);

%% 2. Parameter space
ne_list = linspace(1.4e19, 1.5e19, 10);
d_list = linspace(0.20, 0.30, 20);

num_curves = length(ne_list) * length(d_list);
Y_curves = zeros(num_curves, length(f_probe));
params_map = zeros(num_curves, 3); % [ne, d, fp]

curve_idx = 0;
for ne = ne_list
    for d = d_list
        curve_idx = curve_idx + 1;
        fp = sqrt(ne * e_charge^2 / (eps0 * me)) / (2 * pi);
        params_map(curve_idx, :) = [ne, d, fp];
        Y_curves(curve_idx, :) = calculate_theoretical_delay( ...
            f_probe, ne, nu_val, d, c, eps0, me, e_charge);
    end
end

%% 3. Plot
fig = figure('Name', 'multisolution_analysis', 'Color', 'w', 'Position', [150, 100, 700, 500]);
ax = axes('Parent', fig);
hold(ax, 'on'); grid(ax, 'on');

% Background trajectories
plot(ax, f_probe / 1e9, Y_curves * 1e9, ...
    'Color', [0.5, 0.5, 0.5, 0.15], 'LineWidth', 1.0);

%% 4. Intersection extraction
fprintf('【步骤4】正在计算交点...\n');
cross_count = 0;

for i = 1:num_curves - 1
    for j = i + 1:num_curves
        y1 = Y_curves(i, :);
        y2 = Y_curves(j, :);
        diff_y = y1 - y2;
        idx_cross = find(diff_y(1:end-1) .* diff_y(2:end) < 0);

        for k = 1:length(idx_cross)
            idx = idx_cross(k);
            r = abs(diff_y(idx)) / (abs(diff_y(idx)) + abs(diff_y(idx + 1)));
            f_cross = f_probe(idx) + r * (f_probe(idx + 1) - f_probe(idx));
            tau1 = y1(idx) + r * (y1(idx + 1) - y1(idx));
            tau2 = y2(idx) + r * (y2(idx + 1) - y2(idx));
            tau_cross = (tau1 + tau2) / 2;

            fp_max = max(params_map(i, 3), params_map(j, 3));
            if f_cross <= fp_max
                continue;
            end

            p1 = params_map(i, 1:2);
            p2 = params_map(j, 1:2);
            dn = abs(p1(1) - p2(1)) / mean([p1(1), p2(1)]);
            dd = abs(p1(2) - p2(2)) / mean([p1(2), p2(2)]);

            if dn > 0.05 && dd > 0.05
                plot(ax, f_cross / 1e9, tau_cross * 1e9, 'ro', ...
                    'MarkerFaceColor', 'r', ...
                    'MarkerEdgeColor', 'none', ...
                    'MarkerSize', 4);
                cross_count = cross_count + 1;
            end
        end
    end
end
fprintf('【步骤4】共标记 %d 个物理有效交点\n', cross_count);

%% 5. Styling and legend
xlabel(ax, '探测频率 \fontname{Times New Roman}(GHz)', ...
    'FontSize', 11, 'Interpreter', 'tex');
ylabel(ax, '相对群时延 \fontname{Times New Roman}(ns)', ...
    'FontSize', 11, 'Interpreter', 'tex');
xlim(ax, [34, 37.5]);
ylim(ax, [-1, 9]);
set(ax, 'FontSize', 11, 'Layer', 'bottom');

% Dummy handles for a compact legend
h_dummy_line = plot(ax, NaN, NaN, '-', ...
    'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
h_dummy_dot = plot(ax, NaN, NaN, 'ro', ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
lgd = legend(ax, [h_dummy_line, h_dummy_dot], ...
    {'\fontname{SimSun}理论群时延曲线', '\fontname{SimSun}多解性等效交点'}, ...
    'Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
set(lgd, 'Box', 'off', 'Color', 'none');

%% 6. Export
export_thesis_figure(fig, 'fig_3_8_multisolution_intersections', 14, 600, 'SimSun');

%% 7. Local function
function tau_rel = calculate_theoretical_delay(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
omega = 2 * pi * f_vec;
wp = sqrt(ne_val * e_charge^2 / (eps0 * me));

eps_r = 1 - (wp^2) ./ (omega .* (omega + 1i * nu_val));
k = (omega ./ c) .* sqrt(eps_r);
phi = -real(k) * d;

dphi = diff(phi);
domega = diff(omega);
tau = -dphi ./ domega;
tau = [tau, tau(end)];

tau_rel = tau - d / c;
end
