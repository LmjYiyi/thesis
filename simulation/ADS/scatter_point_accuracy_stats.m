%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 时延特征点提取精度的定量评估
% 功能：将 ESPRIT 提取的离散散点与 ADS S 参数群时延真值逐点对比，
%       按频率分区计算 MAE、RMSE、最大偏差等统计指标
% 依赖：hunpin_time_v.txt (ADS 仿真中频信号), delay.txt (ADS 真值群时延)
% 输出：论文表 5-4 所需的分区精度统计数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

fprintf('======================================================\n');
fprintf('  时延特征点提取精度定量评估\n');
fprintf('  散点来源: 滑动窗口 ESPRIT + 三重物理约束清洗\n');
fprintf('  真值基准: ADS S 参数群时延 (delay.txt)\n');
fprintf('======================================================\n\n');

%% 1. 嵌入的数据提取模块 (与 mcmc_bayesian_inversion.m 完全一致)
% -------------------------------------------------------------------------
data = readmatrix('hunpin_time_v.txt', 'FileType', 'text', 'NumHeaderLines', 1);
valid = ~isnan(data(:,1)) & ~isnan(data(:,2));
t_raw = data(valid, 1); v_raw = data(valid, 2);

T_m = t_raw(end) - t_raw(1);
f_start = 34.4e9; f_end = 37.61e9; 
K = (f_end - f_start) / T_m;          
baseline_delay = 0.2470e-9; 

fs_dec = 4e9; 
t_dec = linspace(t_raw(1), t_raw(end), round(T_m * fs_dec)).';
v_dec = interp1(t_raw, v_raw, t_dec, 'spline');
[b_lp, a_lp] = butter(4, 200e6 / (fs_dec / 2));
s_if = filtfilt(b_lp, a_lp, v_dec);
s_proc = s_if(1:2:end); t_proc = t_dec(1:2:end); f_s_proc = fs_dec / 2;   

win_len = max(round(0.03 * length(s_proc)), 64);
step_len = max(round(win_len / 8), 1);    
L_sub = round(win_len / 2);            

f_probe = []; tau_est = []; amp_est = [];
rms_threshold = max(abs(s_proc)) * 0.005; 
num_windows = floor((length(s_proc) - win_len) / step_len) + 1;

for i = 1:num_windows
    idx = (i-1)*step_len+1 : (i-1)*step_len+win_len;
    if idx(end) > length(s_proc), break; end
    
    x_win = s_proc(idx);
    t_c = t_proc(idx(round(win_len/2)));
    
    if t_c > 0.99*T_m || t_c < 0.01*T_m || rms(x_win) < rms_threshold, continue; end
    
    M_sub = win_len - L_sub + 1;
    X_h = zeros(L_sub, M_sub); for k=1:M_sub, X_h(:,k)=x_win(k:k+L_sub-1).'; end
    R_x = ((X_h*X_h')/M_sub + fliplr(eye(L_sub))*conj((X_h*X_h')/M_sub)*fliplr(eye(L_sub)))/2;
    [V, D] = eig(R_x); [lam, id] = sort(diag(D), 'descend'); V = V(:,id);
    mdl = zeros(length(lam),1);
    for k=0:length(lam)-1
        ns = lam(k+1:end); ns(ns<1e-30)=1e-30;
        mdl(k+1) = -(length(lam)-k)*M_sub*log(prod(ns)^(1/length(ns))/mean(ns)) + 0.5*k*(2*length(lam)-k)*log(M_sub);
    end
    [~, k_est] = min(mdl); num_s = min(max(1, k_est-1), 3);
    
    Us = V(:,1:num_s);
    est_f = abs(angle(eig((Us(1:end-1,:)'*Us(1:end-1,:))\(Us(1:end-1,:)'*Us(2:end,:))))) * f_s_proc/(2*pi);
    est_f = est_f(est_f > 50e3 & est_f < f_s_proc/4);
    
    if ~isempty(est_f)
        calibrated_tau = min(est_f)/K - baseline_delay;
        f_probe = [f_probe, f_start + K*t_c];
        tau_est = [tau_est, calibrated_tau];
        amp_est = [amp_est, rms(x_win)];
    end
end

% 三重物理约束清洗 (与 mcmc_bayesian_inversion.m 一致)
mask_amp = amp_est > max(amp_est) * 0.20; 
mask_physics = tau_est > 1.85e-9; 
valid_mask = mask_amp & mask_physics;

X_fit = f_probe(valid_mask);   % 有效散点频率 (Hz)
Y_fit = tau_est(valid_mask);   % 有效散点时延 (s)
W_raw = amp_est(valid_mask);   % 有效散点 RMS 幅度

fprintf('ESPRIT 特征提取完成，有效散点数: %d\n', length(X_fit));
fprintf('频率范围: %.2f - %.2f GHz\n', min(X_fit)/1e9, max(X_fit)/1e9);
fprintf('时延范围: %.4f - %.4f ns\n\n', min(Y_fit)*1e9, max(Y_fit)*1e9);

%% 2. 加载 ADS 群时延真值曲线 (delay.txt)
% -------------------------------------------------------------------------
delay_data = readmatrix('delay.txt', 'FileType', 'text', 'NumHeaderLines', 1);
f_true = delay_data(:, 1);     % 频率 (Hz)
tau_true = delay_data(:, 2);   % 群时延真值 (s)

fprintf('ADS 真值曲线加载成功，数据点数: %d\n', length(f_true));
fprintf('频率范围: %.2f - %.2f GHz\n\n', min(f_true)/1e9, max(f_true)/1e9);

%% 3. 逐点对比：在散点频率处插值获取真值
% -------------------------------------------------------------------------
tau_true_at_scatter = interp1(f_true, tau_true, X_fit, 'spline');

% 提取残差 (散点 - 真值)
residuals = Y_fit - tau_true_at_scatter;          % 单位: s
abs_residuals = abs(residuals);                    % 绝对残差

fprintf('======================================================\n');
fprintf('  全频段综合精度统计 (共 %d 个散点)\n', length(X_fit));
fprintf('======================================================\n');
fprintf('  MAE  (平均绝对误差):   %.4f ns\n', mean(abs_residuals)*1e9);
fprintf('  RMSE (均方根误差):     %.4f ns\n', sqrt(mean(residuals.^2))*1e9);
fprintf('  最大偏差 (Max |Δτ|):   %.4f ns\n', max(abs_residuals)*1e9);
fprintf('  平均偏置 (Mean Δτ):    %.4f ns\n', mean(residuals)*1e9);
fprintf('  标准差 (Std Δτ):       %.4f ns\n\n', std(residuals)*1e9);

%% 4. 分区精度统计 (论文表 5-4)
% -------------------------------------------------------------------------
% 频率分区定义 (单位: GHz → Hz)
% 通带平坦区:    36.7 ~ 37.3 GHz (群时延梯度 ≤ 5 ns/GHz)
% 色散过渡区:    36.5~36.7 GHz 和 37.3~37.5 GHz
% 色散双峰陡变区: <36.5 GHz 和 >37.5 GHz (在有效散点范围内)

X_GHz = X_fit / 1e9;  % 转换为 GHz 便于分区判定

% 定义分区
mask_flat = (X_GHz >= 36.7) & (X_GHz <= 37.3);
mask_transition = ((X_GHz >= 36.5) & (X_GHz < 36.7)) | ...
                  ((X_GHz > 37.3) & (X_GHz <= 37.5));
mask_peak = ((X_GHz >= 36.43) & (X_GHz < 36.5)) | (X_GHz > 37.5);

% 兜底：未被以上三类覆盖的散点归入最近的分区
mask_unclassified = ~mask_flat & ~mask_transition & ~mask_peak;
if any(mask_unclassified)
    fprintf('[警告] 存在 %d 个散点未被三个分区覆盖，将归入色散过渡区\n', sum(mask_unclassified));
    mask_transition = mask_transition | mask_unclassified;
end

% 计算各分区统计量
zones = {'通带平坦区 (36.7~37.3 GHz)', '色散过渡区 (过渡带)', '色散双峰陡变区 (峰顶附近)'};
masks = {mask_flat, mask_transition, mask_peak};

fprintf('======================================================\n');
fprintf('  分区精度统计 (论文表 5-4)\n');
fprintf('======================================================\n');
fprintf('%-35s | 散点数 | MAE(ns) | RMSE(ns) | MaxDev(ns) | 相对误差*\n', '频率分区');
fprintf('%s\n', repmat('-', 1, 100));

all_zone_data = {};  % 用于后续汇总输出

for z = 1:length(zones)
    m = masks{z};
    n_pts = sum(m);
    
    if n_pts == 0
        fprintf('%-35s | %6d | %7s | %8s | %10s | %s\n', zones{z}, 0, 'N/A', 'N/A', 'N/A', 'N/A');
        all_zone_data{z} = struct('name', zones{z}, 'n', 0, 'mae', NaN, 'rmse', NaN, 'maxdev', NaN, 'rel_err', NaN);
        continue;
    end
    
    res_zone = residuals(m);
    abs_res_zone = abs_residuals(m);
    tau_true_zone = tau_true_at_scatter(m);
    
    mae_zone = mean(abs_res_zone) * 1e9;
    rmse_zone = sqrt(mean(res_zone.^2)) * 1e9;
    maxdev_zone = max(abs_res_zone) * 1e9;
    
    % 相对误差 = MAE / 该分区理论群时延均值
    mean_tau_true_zone = mean(abs(tau_true_zone)) * 1e9;
    rel_err_zone = mae_zone / mean_tau_true_zone * 100;
    
    fprintf('%-35s | %6d | %7.4f | %8.4f | %10.4f | %6.2f%%\n', ...
        zones{z}, n_pts, mae_zone, rmse_zone, maxdev_zone, rel_err_zone);
    
    all_zone_data{z} = struct('name', zones{z}, 'n', n_pts, 'mae', mae_zone, ...
        'rmse', rmse_zone, 'maxdev', maxdev_zone, 'rel_err', rel_err_zone);
end

% 全频段汇总行
fprintf('%s\n', repmat('-', 1, 100));
mae_all = mean(abs_residuals) * 1e9;
rmse_all = sqrt(mean(residuals.^2)) * 1e9;
maxdev_all = max(abs_residuals) * 1e9;
mean_tau_true_all = mean(abs(tau_true_at_scatter)) * 1e9;
rel_err_all = mae_all / mean_tau_true_all * 100;
fprintf('%-35s | %6d | %7.4f | %8.4f | %10.4f | %6.2f%%\n', ...
    '全频段综合', length(X_fit), mae_all, rmse_all, maxdev_all, rel_err_all);
fprintf('======================================================\n\n');

% *注释
fprintf('* 相对误差定义: MAE / 该分区理论群时延均值\n\n');

%% 5. 逐点残差明细表
% -------------------------------------------------------------------------
fprintf('======================================================\n');
fprintf('  逐点残差明细 (供核验)\n');
fprintf('======================================================\n');
fprintf('%5s | %12s | %12s | %12s | %12s | %8s\n', ...
    '序号', '频率(GHz)', '散点τ(ns)', '真值τ(ns)', '残差Δτ(ns)', '分区');

for i = 1:length(X_fit)
    % 判断分区
    if mask_flat(i)
        zone_label = '平坦区';
    elseif mask_transition(i)
        zone_label = '过渡区';
    elseif mask_peak(i)
        zone_label = '峰顶区';
    else
        zone_label = '未分类';
    end
    
    fprintf('%5d | %12.4f | %12.4f | %12.4f | %+12.4f | %8s\n', ...
        i, X_fit(i)/1e9, Y_fit(i)*1e9, tau_true_at_scatter(i)*1e9, ...
        residuals(i)*1e9, zone_label);
end
fprintf('======================================================\n\n');

%% 6. 可视化：散点与真值对比 + 残差分布
% -------------------------------------------------------------------------

% --- 图1: 散点与真值叠加 + 分区标注 ---
figure('Color', 'w', 'Position', [100, 100, 900, 550]);
hold on;

% ADS 真值连续曲线
plot(f_true/1e9, tau_true*1e9, '-', 'Color', [0.1 0.6 0.1], 'LineWidth', 2.5, ...
    'DisplayName', 'ADS 群时延真值 \tau_g^{true}(f)');

% 分区底色
ylims = [0, 9];
fill([36.7 37.3 37.3 36.7], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [0.9 0.95 1.0], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '通带平坦区');
fill([36.5 36.7 36.7 36.5], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1.0 0.95 0.85], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([37.3 37.5 37.5 37.3], [ylims(1) ylims(1) ylims(2) ylims(2)], ...
    [1.0 0.95 0.85], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '色散过渡区');

% 散点 (颜色编码权重)
Weights = (W_raw / max(W_raw)).^2;
scatter(X_fit/1e9, Y_fit*1e9, 60, Weights, 'filled', 'MarkerEdgeColor', 'k', ...
    'LineWidth', 0.5, 'DisplayName', 'ESPRIT 提取散点');
cb = colorbar; ylabel(cb, '归一化权重 w_k', 'FontSize', 11);

grid on; set(gca, 'GridAlpha', 0.3, 'FontSize', 11);
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('群时延 \tau_g (ns)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([36.3, 37.7]); ylim(ylims);
title('散点与 ADS 真值逐点对比 (分区标注)', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 10);

% --- 图2: 残差分布图 ---
figure('Color', 'w', 'Position', [150, 150, 900, 450]);

subplot(1,2,1);
hold on;
% 按分区绘制残差散点
colors = {[0.2 0.6 0.8], [0.9 0.6 0.1], [0.8 0.2 0.2]};
zone_labels_short = {'平坦区', '过渡区', '峰顶区'};
for z = 1:length(masks)
    m = masks{z};
    if sum(m) > 0
        scatter(X_fit(m)/1e9, residuals(m)*1e9, 50, colors{z}, 'filled', ...
            'DisplayName', zone_labels_short{z});
    end
end
yline(0, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
yline(0.17, 'r:', 'RMSE', 'LineWidth', 1, 'HandleVisibility', 'off');
yline(-0.17, 'r:', '', 'LineWidth', 1, 'HandleVisibility', 'off');
grid on;
xlabel('瞬时探测频率 (GHz)', 'FontSize', 12);
ylabel('残差 \Delta\tau (ns)', 'FontSize', 12);
title('(a) 逐点残差 vs 频率', 'FontSize', 13);
legend('Location', 'best', 'FontSize', 10);

subplot(1,2,2);
histogram(residuals*1e9, 15, 'Normalization', 'pdf', 'FaceColor', [0.3 0.5 0.7], 'EdgeAlpha', 0.3);
hold on;
xline(0, 'k--', 'LineWidth', 1);
xline(mean(residuals)*1e9, 'r-', 'LineWidth', 2);
grid on;
xlabel('残差 \Delta\tau (ns)', 'FontSize', 12);
ylabel('概率密度', 'FontSize', 12);
title('(b) 残差直方图', 'FontSize', 13);
legend({'残差分布', '零基线', sprintf('均值 = %.4f ns', mean(residuals)*1e9)}, ...
    'Location', 'best', 'FontSize', 10);

sgtitle('时延特征点提取精度：残差分析', 'FontSize', 15, 'FontWeight', 'bold');

%% 7. 输出论文表格格式 (可直接复制至 Markdown)
% -------------------------------------------------------------------------
fprintf('\n======================================================\n');
fprintf('  论文 Markdown 表格输出 (可直接复制)\n');
fprintf('======================================================\n\n');

fprintf('**表5-4** 时延特征点提取精度统计（与ADS S参数真值逐点对比）\n\n');
fprintf('| 频率分区 | 散点数 | MAE (ns) | RMSE (ns) | 最大偏差 (ns) | 相对误差* |\n');
fprintf('|:----:|:----:|:----:|:----:|:----:|:----:|\n');

for z = 1:length(all_zone_data)
    d = all_zone_data{z};
    if d.n > 0
        fprintf('| %s | %d | %.2f | %.2f | %.2f | %.1f%% |\n', ...
            d.name, d.n, d.mae, d.rmse, d.maxdev, d.rel_err);
    end
end
fprintf('| **全频段综合** | **%d** | **%.2f** | **%.2f** | **%.2f** | **%.1f%%** |\n', ...
    length(X_fit), mae_all, rmse_all, maxdev_all, rel_err_all);
fprintf('\n*注：相对误差定义为MAE与该分区理论群时延均值之比。\n');

fprintf('\n======================================================\n');
fprintf('  精度评估输出完毕\n');
fprintf('======================================================\n');
