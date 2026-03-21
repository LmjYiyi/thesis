function print_s2p_deviation_report(scatter_in, cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print point-to-s2p deviation report for trajectory optimization.
% Deviation definition: delta_tau = tau_point - tau_s2p(freq_point).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(cfg, 'deviation_report') || ~isfield(cfg.deviation_report, 'enable') ...
        || ~cfg.deviation_report.enable
    return;
end

if ~isfield(scatter_in, 'f_probe') || ~isfield(scatter_in, 'tau') ...
        || isempty(scatter_in.f_probe) || isempty(scatter_in.tau)
    fprintf('\n[s2p deviation] no valid scatter points.\n');
    return;
end

top_n = 8;
if isfield(cfg.deviation_report, 'top_n') && ~isempty(cfg.deviation_report.top_n)
    top_n = max(1, round(cfg.deviation_report.top_n));
end

lib_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(lib_dir);
s2p_file = fullfile(root_dir, 'data', 'HXLBQ-DTA1329-1-1.s2p');
[f_s2p_hz, tau_s2p_s] = local_read_s21_group_delay_from_s2p(s2p_file);
if isempty(f_s2p_hz) || isempty(tau_s2p_s)
    warning('print_s2p_deviation_report: cannot read s2p curve: %s', s2p_file);
    return;
end

f_min = min(f_s2p_hz);
f_max = max(f_s2p_hz);
mask_cmp = isfinite(scatter_in.f_probe) & isfinite(scatter_in.tau) & ...
    scatter_in.f_probe >= f_min & scatter_in.f_probe <= f_max;

if ~any(mask_cmp)
    fprintf('\n[s2p deviation] no scatter points within s2p frequency range.\n');
    return;
end

f_cmp_hz = scatter_in.f_probe(mask_cmp);
tau_cmp_s = scatter_in.tau(mask_cmp);
tau_ref_s = interp1(f_s2p_hz, tau_s2p_s, f_cmp_hz, 'pchip');
mask_ok = isfinite(tau_ref_s);

if ~any(mask_ok)
    fprintf('\n[s2p deviation] interpolation failed on all comparable points.\n');
    return;
end

f_cmp_hz = f_cmp_hz(mask_ok);
tau_cmp_s = tau_cmp_s(mask_ok);
tau_ref_s = tau_ref_s(mask_ok);
delta_ns = (tau_cmp_s - tau_ref_s) * 1e9;

src_code = [];
if isfield(scatter_in, 'source_code') && numel(scatter_in.source_code) == numel(scatter_in.f_probe)
    src_code = scatter_in.source_code(mask_cmp);
    src_code = src_code(mask_ok);
end

fprintf('\n[s2p deviation] delta_tau = tau_point - tau_s2p\n');
fprintf('  comparable points: %d, freq range: %.3f-%.3f GHz\n', ...
    numel(delta_ns), min(f_cmp_hz) / 1e9, max(f_cmp_hz) / 1e9);
print_one_region_summary('all points', f_cmp_hz, delta_ns);

if isfield(cfg, 'cfg_hybrid') && isfield(cfg.cfg_hybrid, 'f_flat_hi')
    mask_right = f_cmp_hz > cfg.cfg_hybrid.f_flat_hi;
    if any(mask_right)
        print_one_region_summary('right edge', f_cmp_hz(mask_right), delta_ns(mask_right));
    end
end

if ~isempty(src_code)
    print_source_summary(src_code, delta_ns);
end

[~, idx_sort] = sort(abs(delta_ns), 'descend');
n_show = min(top_n, numel(idx_sort));
fprintf('  top-%d |delta| points:\n', n_show);
fprintf('    %10s  %10s  %10s  %10s  %8s  %6s\n', ...
    'f_GHz', 'tau_pt_ns', 'tau_s2p_ns', 'delta_ns', 'sign', 'src');
for k = 1:n_show
    idx = idx_sort(k);
    if isempty(src_code)
        src_label = '-';
    else
        src_label = source_code_label(src_code(idx));
    end

    if delta_ns(idx) > 0
        sign_label = 'above';
    elseif delta_ns(idx) < 0
        sign_label = 'below';
    else
        sign_label = 'equal';
    end

    fprintf('    %10.4f  %10.4f  %10.4f  %+10.4f  %8s  %6s\n', ...
        f_cmp_hz(idx) / 1e9, tau_cmp_s(idx) * 1e9, tau_ref_s(idx) * 1e9, ...
        delta_ns(idx), sign_label, src_label);
end

end

function print_one_region_summary(region_name, f_hz, delta_ns)
n_pos = sum(delta_ns > 0);
n_neg = sum(delta_ns < 0);
n_zero = sum(delta_ns == 0);
[delta_max, idx_max] = max(delta_ns);
[delta_min, idx_min] = min(delta_ns);

fprintf('  [%s]\n', region_name);
fprintf('    pos/neg/zero: %d / %d / %d\n', n_pos, n_neg, n_zero);
fprintf('    mean = %+0.4f ns, mean|.| = %.4f ns, rms = %.4f ns\n', ...
    mean(delta_ns), mean(abs(delta_ns)), sqrt(mean(delta_ns .^ 2)));
fprintf('    max positive: %+0.4f ns @ %.4f GHz\n', delta_max, f_hz(idx_max) / 1e9);
fprintf('    max negative: %+0.4f ns @ %.4f GHz\n', delta_min, f_hz(idx_min) / 1e9);
end

function print_source_summary(src_code, delta_ns)
codes = unique(src_code(:)).';
fprintf('  by source:\n');
for code = codes
    mask = src_code == code;
    fprintf('    %-6s n=%-3d mean=%+0.4f ns mean|.|=%.4f ns\n', ...
        source_code_label(code), sum(mask), mean(delta_ns(mask)), mean(abs(delta_ns(mask))));
end
end

function label = source_code_label(code)
switch code
    case 1
        label = 'base';
    case 2
        label = 'adapt';
    case 3
        label = 'dense';
    otherwise
        label = sprintf('s%d', code);
end
end

function [f_hz, tau_s] = local_read_s21_group_delay_from_s2p(s2p_file)
fid = fopen(s2p_file, 'r');
if fid < 0
    f_hz = [];
    tau_s = [];
    return;
end

cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

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
