function out = likely_filter_delay_curve(scatter_in, script_dir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build group-delay reference curve for plotting.
% Primary source: measured Touchstone file data/HXLBQ-DTA1329-1-1.s2p.
% Fallback source: previous analytic/template "likely" curve when s2p
% parsing fails or valid passband samples are insufficient.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_lo = 36.5e9;
f_hi = 37.5e9;
f0 = 37.0e9;
s2p_file = fullfile(script_dir, 'data', 'HXLBQ-DTA1329-1-1.s2p');

[f_plot, tau_ns, ok_s2p] = build_curve_from_s2p(s2p_file, f_lo, f_hi);
if ~ok_s2p
    [f_plot, tau_ns] = build_legacy_curve(scatter_in, script_dir, f_lo, f_hi, f0);
end

mask_mid = f_plot >= 36.82e9 & f_plot <= 37.18e9;
if any(mask_mid)
    tau_floor_ns = median(tau_ns(mask_mid));
else
    tau_floor_ns = min(tau_ns);
end

[tau_peak_ns, idx_peak] = max(tau_ns);
out.f_hz = f_plot(:);
out.f_ghz = f_plot(:) / 1e9;
out.tau_ns = tau_ns(:);
out.tau_floor_ns = tau_floor_ns;
out.tau_peak_ns = tau_peak_ns;
out.f0_ghz = f_plot(idx_peak) / 1e9;
end

function [f_plot, tau_ns, ok] = build_curve_from_s2p(s2p_file, f_lo, f_hi)
f_plot = linspace(f_lo, f_hi, 401).';
tau_ns = [];
ok = false;

if ~exist(s2p_file, 'file')
    return;
end

[f_hz, tau_s] = read_s21_group_delay_from_s2p(s2p_file);
if numel(f_hz) < 10 || numel(tau_s) ~= numel(f_hz)
    return;
end

mask_band = f_hz >= f_lo & f_hz <= f_hi & isfinite(tau_s);
if sum(mask_band) < 20
    return;
end

f_band = f_hz(mask_band);
tau_band_ns = tau_s(mask_band) * 1e9;

[f_band, ia] = unique(f_band, 'stable');
tau_band_ns = tau_band_ns(ia);
if numel(f_band) < 20
    return;
end

tau_band_ns = smoothdata(tau_band_ns, 'movmedian', 7);
tau_band_ns = smoothdata(tau_band_ns, 'movmean', 7);

tau_ns = interp1(f_band, tau_band_ns, f_plot, 'pchip', 'extrap');
p_low = percentile_linear(tau_band_ns, 0.02);
p_high = percentile_linear(tau_band_ns, 0.98);
tau_ns = min(max(tau_ns, p_low), p_high);

ok = all(isfinite(tau_ns));
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
    otherwise % DB
        s21 = 10 .^ (s21_a / 20) .* exp(1i * deg2rad(s21_b));
end

phase_rad = unwrap(angle(s21));
tau_s = -gradient(phase_rad, f_hz) / (2 * pi);

mask_ok = isfinite(f_hz) & isfinite(tau_s);
f_hz = f_hz(mask_ok);
tau_s = tau_s(mask_ok);
end

function [f_plot, tau_ns] = build_legacy_curve(scatter_in, script_dir, f_lo, f_hi, f0)
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

function val = percentile_linear(x, q)
x = x(:);
x = x(isfinite(x));
if isempty(x)
    val = NaN;
    return;
end

x = sort(x);
q = min(max(q, 0), 1);
pos = 1 + (numel(x) - 1) * q;
i_lo = floor(pos);
i_hi = ceil(pos);

if i_lo == i_hi
    val = x(i_lo);
else
    w = pos - i_lo;
    val = (1 - w) * x(i_lo) + w * x(i_hi);
end
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
