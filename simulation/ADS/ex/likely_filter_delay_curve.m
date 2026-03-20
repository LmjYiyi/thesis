function out = likely_filter_delay_curve(scatter_in, script_dir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build a likely cavity-filter group-delay curve constrained by:
% 1. Product inspection record: passband 36.5-37.5 GHz, F0 = 37.0 GHz
% 2. Measured group delay figure: 1.97 ns
% 3. Extracted scatter: used only to anchor the mid-band baseline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_lo = 36.5e9;
f_hi = 37.5e9;
f0 = 37.0e9;
tau_peak_ns = 1.97;
tau_limit_ns = 2.30;

f_plot = linspace(f_lo, f_hi, 401).';
shape = build_shape_template(f_plot, f0, f_lo, f_hi, script_dir);

mask_mid = scatter_in.f_probe >= 36.82e9 & scatter_in.f_probe <= 37.18e9;
if any(mask_mid)
    tau_floor_ns = median(scatter_in.tau(mask_mid)) * 1e9;
else
    tau_floor_ns = 1.55;
end

tau_floor_ns = max(tau_floor_ns, 1.35);
tau_floor_ns = min(tau_floor_ns, tau_peak_ns - 0.18);

tau_ns = tau_floor_ns + (tau_peak_ns - tau_floor_ns) * shape;
tau_ns = min(tau_ns, tau_limit_ns);

out.f_hz = f_plot;
out.f_ghz = f_plot / 1e9;
out.tau_ns = tau_ns;
out.tau_floor_ns = tau_floor_ns;
out.tau_peak_ns = max(tau_ns);
out.f0_ghz = f0 / 1e9;
end

function shape = build_shape_template(f_plot, f0, f_lo, f_hi, script_dir)
x = abs((f_plot - f0) / (0.5 * (f_hi - f_lo)));
x = min(max(x, 0), 1);
shape_analytic = sin(0.5 * pi * x) .^ 1.6;

shape = shape_analytic;

root_dir = fileparts(fileparts(fileparts(script_dir)));
template_file = fullfile(root_dir, 'thesis-code', 'cst_filter', 'groupdelay.txt');
if ~exist(template_file, 'file')
    return;
end

[f_template_hz, tau_template_ns] = read_template_curve(template_file);
if numel(f_template_hz) < 10
    return;
end

tau_left = interp1(f_template_hz, tau_template_ns, f_plot, 'pchip', NaN);
tau_right = interp1(f_template_hz, tau_template_ns, 2 * f0 - f_plot, 'pchip', NaN);
mask_ok = isfinite(tau_left) & isfinite(tau_right);
if sum(mask_ok) < round(0.8 * numel(f_plot))
    return;
end

tau_sym = 0.5 * (tau_left(mask_ok) + tau_right(mask_ok));
tau_sym = movmean(tau_sym, 21);
shape_template = normalize01(tau_sym);
shape_template = shape_template .^ 0.85;

shape(mask_ok) = 0.70 * shape_template + 0.30 * shape_analytic(mask_ok);
shape = normalize01(shape);
end

function [f_hz, tau_ns] = read_template_curve(template_file)
fid = fopen(template_file, 'r');
if fid < 0
    f_hz = [];
    tau_ns = [];
    return;
end

cleaner = onCleanup(@() fclose(fid));
data = textscan(fid, '%f%f', 'Delimiter', '\t', 'CommentStyle', '#');
f_hz = data{1} * 1e9;
tau_ns = data{2};

mask = isfinite(f_hz) & isfinite(tau_ns);
f_hz = f_hz(mask);
tau_ns = tau_ns(mask);
end

function y = normalize01(x)
x = x(:);
x_min = min(x);
x_max = max(x);
if ~isfinite(x_min) || ~isfinite(x_max) || x_max <= x_min
    y = zeros(size(x));
else
    y = (x - x_min) / (x_max - x_min);
end
end
