%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 三组实测数据质量对比
% 对比: lowpassfilter_filter.csv / _1.csv / _2.csv
% 指标: 原始幅度、叠加平均后幅度、频谱 SNR、主峰频率
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
out_dir = fullfile(project_root, 'figures_export');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

files = {fullfile(script_dir, 'data', 'lowpassfilter_filter.csv'), ...
         fullfile(script_dir, 'data', 'lowpassfilter_filter_1.csv'), ...
         fullfile(script_dir, 'data', 'lowpassfilter_filter_2.csv')};
labels = {'原始', '新1', '新2'};

T_m = 50e-6;

fprintf('============================================================\n');
fprintf('           三组实测数据质量对比\n');
fprintf('============================================================\n\n');

%% 1. 基本统计量
fprintf('--- 1. 基本统计 ---\n');
fprintf('  %-6s  %-10s  %-10s  %-12s  %-12s  %-10s\n', ...
    '数据', 'fs(MHz)', 'Vpp原始(mV)', 'Vrms原始(mV)', 'Vpp叠加(mV)', 'Vrms叠加(mV)');

v_avg_all = cell(3,1);
fs_all = zeros(3,1);

for ii = 1:3
    data = readmatrix(files{ii});
    t = data(:,1); v = data(:,2);
    dt = median(diff(t));
    fs = round(1/dt);
    fs_all(ii) = fs;
    N_total = length(v);
    N_per = round(T_m * fs);
    N_sweep = floor(N_total / N_per);

    v = v - mean(v);
    Vpp_raw = max(v) - min(v);
    Vrms_raw = rms(v);

    v_mat = reshape(v(1:N_sweep*N_per), N_per, N_sweep);
    v_avg = mean(v_mat, 2);
    v_avg_all{ii} = v_avg;
    Vpp_avg = max(v_avg) - min(v_avg);
    Vrms_avg = rms(v_avg);

    fprintf('  %-6s  %-10.0f  %-10.3f  %-12.4f  %-12.4f  %-12.5f\n', ...
        labels{ii}, fs/1e6, Vpp_raw*1e3, Vrms_raw*1e3, Vpp_avg*1e3, Vrms_avg*1e3);
end

%% 2. 叠加平均后频谱对比
fprintf('\n--- 2. 频谱分析 (叠加平均 + 降采样 20MHz 后) ---\n');
fprintf('  %-6s  %-12s  %-12s  %-12s  %-12s\n', ...
    '数据', '主峰(kHz)', '峰值(dB)', '噪底(dB)', '频谱SNR(dB)');

figure('Color', 'w', 'Position', [50, 50, 1400, 400]);
colors = {'b', 'r', [0 0.6 0]};

for ii = 1:3
    v_avg = v_avg_all{ii};
    fs = fs_all(ii);

    % 降采样到 20 MHz
    ds = 10;
    v_ds = v_avg(1:ds:end);
    fs_ds = fs / ds;
    N_ds = length(v_ds);

    % FFT
    nfft = 2^nextpow2(N_ds);
    S = abs(fft(v_ds, nfft));
    S = S(1:nfft/2+1);
    S_dB = 20*log10(S / max(S) + eps);
    f_axis = (0:nfft/2) * fs_ds / nfft;

    subplot(1, 3, ii);
    plot(f_axis/1e3, S_dB, 'Color', colors{ii}, 'LineWidth', 1.2);
    xlim([0, 500]); ylim([-80, 5]);
    grid on;
    xlabel('频率 (kHz)', 'FontSize', 11);
    ylabel('归一化幅度 (dB)', 'FontSize', 11);
    title(sprintf('数据 %s', labels{ii}), 'FontSize', 12);
    set(gca, 'FontName', 'SimHei', 'FontSize', 10);

    % 主峰
    search_mask = f_axis > 20e3 & f_axis < 300e3;
    S_search = S; S_search(~search_mask) = 0;
    [S_peak, idx_peak] = max(S_search);
    f_peak = f_axis(idx_peak);

    % 噪底 (200-400 kHz 区间中值)
    noise_mask = f_axis > 200e3 & f_axis < 400e3;
    noise_floor = median(S(noise_mask));
    snr_spec = 20*log10(S_peak / (noise_floor + eps));

    fprintf('  %-6s  %-12.1f  %-12.1f  %-12.1f  %-12.1f\n', ...
        labels{ii}, f_peak/1e3, 20*log10(S_peak/max(S)+eps), ...
        20*log10(noise_floor/max(S)+eps), snr_spec);

    % 在图上标注主峰
    hold on;
    plot(f_peak/1e3, S_dB(idx_peak), 'v', 'MarkerSize', 8, ...
        'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    text(f_peak/1e3 + 15, S_dB(idx_peak), ...
        sprintf('%.0f kHz\nSNR=%.0f dB', f_peak/1e3, snr_spec), ...
        'FontSize', 9, 'FontName', 'SimHei');
    hold off;
end

sgtitle('三组实测数据叠加平均后频谱对比', 'FontSize', 13, 'FontName', 'SimHei');
file_quality = fullfile(project_root, 'figures_export', 'data_quality_compare.tiff');
exportgraphics(gcf, file_quality, 'Resolution', 200);
fprintf('\n  频谱对比图已导出: %s\n', file_quality);

%% 3. 时域波形对比 (叠加平均后前 200 点)
fprintf('\n--- 3. 时域波形对比 (叠加平均后) ---\n');
figure('Color', 'w', 'Position', [50, 500, 900, 400]);
hold on;
for ii = 1:3
    v_avg = v_avg_all{ii};
    fs = fs_all(ii);
    t_avg = (0:length(v_avg)-1).' / fs * 1e6;  % μs
    plot(t_avg, v_avg * 1e3, 'Color', colors{ii}, 'LineWidth', 1.0);
end
hold off;
grid on;
xlabel('时间 (μs)', 'FontSize', 11);
ylabel('幅度 (mV)', 'FontSize', 11);
title('叠加平均后单周期波形', 'FontSize', 12);
legend(labels, 'Location', 'northeast', 'FontSize', 10);
set(gca, 'FontName', 'SimHei', 'FontSize', 10);
xlim([0, T_m*1e6]);
file_wave = fullfile(project_root, 'figures_export', 'data_waveform_compare.tiff');
exportgraphics(gcf, file_wave, 'Resolution', 200);
fprintf('  波形对比图已导出: %s\n', file_wave);

fprintf('\n============================================================\n');
fprintf('请在 MATLAB 中运行此脚本，然后将输出结果贴回给我。\n');
fprintf('============================================================\n');
