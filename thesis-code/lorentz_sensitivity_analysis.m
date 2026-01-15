%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lorentz模型参数敏感性分析
% 研究课题：谐振频率(f_res)和阻尼因子(γ)对群时延的敏感性
% 创建日期：2026-01-15
% 研究目标：验证Lorentz模型是否能像Drude模型一样进行参数降维
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. 参数设置
fprintf('========================================\n');
fprintf('Lorentz模型参数敏感性分析\n');
fprintf('========================================\n\n');

% 物理常数
c = 3e8;                    % 光速 (m/s)
epsilon_0 = 8.854e-12;      % 真空介电常数

% 探测频率范围
f_probe = linspace(33e9, 38e9, 500);  % 33-38 GHz
omega_probe = 2*pi*f_probe;

% 超材料参数
d = 0.15;                   % 超材料厚度 (m)
omega_p_meta = 2*pi*5e9;    % 等效等离子体频率

% 基准参数
f_res_base = 35.5e9;        % 基准谐振频率 (Hz)
gamma_base = 0.5e9;         % 基准阻尼因子 (Hz)

%% 2. 实验组1：固定γ，改变f_res（验证谐振频率的主导性）

fprintf('=== 实验组1：固定阻尼因子，改变谐振频率 ===\n');

% 设置三组谐振频率（±0.5 GHz变化）
f_res_set1 = [35.0e9, 35.5e9, 36.0e9];  
gamma_fixed = gamma_base;

% 初始化存储
tau_group1 = zeros(3, length(f_probe));
amplitude_dB1 = zeros(3, length(f_probe));

for idx = 1:3
    f_res_val = f_res_set1(idx);
    omega_res = 2*pi*f_res_val;
    
    % 计算Lorentz复介电常数
    epsilon_r = 1 + (omega_p_meta^2) ./ (omega_res^2 - omega_probe.^2 - 1i*2*pi*gamma_fixed*omega_probe);
    
    % 复波数
    k_complex = (omega_probe / c) .* sqrt(epsilon_r);
    
    % 相位
    phase = -real(k_complex) * d;
    
    % 群时延（相位对频率求导）
    tau_total = -gradient(phase, omega_probe);
    tau_group1(idx, :) = tau_total - d/c;  % 相对时延
    
    % 传输幅度
    H = exp(-1i*real(k_complex)*d - abs(imag(k_complex))*d);
    amplitude_dB1(idx, :) = 20*log10(abs(H));
    
    fprintf('  f_res = %.2f GHz, γ = %.2f GHz\n', f_res_val/1e9, gamma_fixed/1e9);
end

%% 3. 实验组2：固定f_res，改变γ（验证阻尼因子的影响）

fprintf('\n=== 实验组2：固定谐振频率，改变阻尼因子 ===\n');

% 设置三组阻尼因子（变化10倍）
gamma_set2 = [0.1e9, 0.5e9, 1.0e9];  
f_res_fixed = f_res_base;

% 初始化存储
tau_group2 = zeros(3, length(f_probe));
amplitude_dB2 = zeros(3, length(f_probe));

for idx = 1:3
    gamma_val = gamma_set2(idx);
    omega_res = 2*pi*f_res_fixed;
    
    % 计算Lorentz复介电常数
    epsilon_r = 1 + (omega_p_meta^2) ./ (omega_res^2 - omega_probe.^2 - 1i*2*pi*gamma_val*omega_probe);
    
    % 复波数
    k_complex = (omega_probe / c) .* sqrt(epsilon_r);
    
    % 相位
    phase = -real(k_complex) * d;
    
    % 群时延
    tau_total = -gradient(phase, omega_probe);
    tau_group2(idx, :) = tau_total - d/c;
    
    % 传输幅度
    H = exp(-1i*real(k_complex)*d - abs(imag(k_complex))*d);
    amplitude_dB2(idx, :) = 20*log10(abs(H));
    
    fprintf('  f_res = %.2f GHz, γ = %.2f GHz\n', f_res_fixed/1e9, gamma_val/1e9);
end

%% 4. 敏感度指标计算

fprintf('\n=== 敏感度指标计算 ===\n');

% 在谐振频率附近选取代表性频点
idx_near_res = find(abs(f_probe - f_res_base) < 0.5e9, 1);  % 35 GHz附近

% 实验组1的敏感度（f_res变化±1.4%，即±0.5GHz）
delta_f_res_percent = (f_res_set1(3) - f_res_set1(1)) / f_res_set1(2);  % 相对变化
delta_tau1 = abs(tau_group1(3, idx_near_res) - tau_group1(1, idx_near_res));
tau1_avg = mean([tau_group1(1, idx_near_res), tau_group1(3, idx_near_res)]);
delta_tau1_percent = delta_tau1 / abs(tau1_avg);

S_tau_fres = delta_tau1_percent / delta_f_res_percent;

fprintf('谐振频率敏感度分析（%+.1f GHz处）：\n', f_probe(idx_near_res)/1e9);
fprintf('  f_res 相对变化: %.2f%%\n', delta_f_res_percent*100);
fprintf('  群时延相对变化: %.2f%%\n', delta_tau1_percent*100);
fprintf('  敏感度指标 S_τ^fres: %.2f\n', S_tau_fres);

% 实验组2的敏感度（γ变化10倍）
delta_gamma_percent = (gamma_set2(3) - gamma_set2(1)) / gamma_set2(2);
delta_tau2 = abs(tau_group2(3, idx_near_res) - tau_group2(1, idx_near_res));
tau2_avg = mean([tau_group2(1, idx_near_res), tau_group2(3, idx_near_res)]);
delta_tau2_percent = delta_tau2 / abs(tau2_avg);

S_tau_gamma = delta_tau2_percent / delta_gamma_percent;

fprintf('\n阻尼因子敏感度分析（%+.1f GHz处）：\n', f_probe(idx_near_res)/1e9);
fprintf('  γ 相对变化: %.2f%%\n', delta_gamma_percent*100);
fprintf('  群时延相对变化: %.2f%%\n', delta_tau2_percent*100);
fprintf('  敏感度指标 S_τ^γ: %.2f\n', S_tau_gamma);

fprintf('\n敏感度比值 S_τ^fres / S_τ^γ = %.2f\n', S_tau_fres / S_tau_gamma);

%% 5. 可视化

% 颜色方案
colors = [
    0.0000, 0.4470, 0.7410;  % 蓝色
    0.8500, 0.3250, 0.0980;  % 橙色
    0.9290, 0.6940, 0.1250;  % 黄色
];

% --- Figure 1: 谐振频率主导性（类比论文图3-3a） ---
figure(1); clf;
set(gcf, 'Position', [100, 100, 900, 700]);

% 子图1：群时延
subplot(2,1,1);
hold on;
for idx = 1:3
    plot(f_probe/1e9, tau_group1(idx, :)*1e9, 'LineWidth', 2.5, 'Color', colors(idx, :), ...
        'DisplayName', sprintf('f_{res} = %.2f GHz', f_res_set1(idx)/1e9));
end
% 标注谐振频率位置
for idx = 1:3
    xline(f_res_set1(idx)/1e9, '--', 'Color', colors(idx, :), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
grid on; box on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相对群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('固定阻尼因子 γ = 0.5 GHz，改变谐振频率', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

% 子图2：透射幅度
subplot(2,1,2);
hold on;
for idx = 1:3
    plot(f_probe/1e9, amplitude_dB1(idx, :), '--', 'LineWidth', 2.5, 'Color', colors(idx, :), ...
        'DisplayName', sprintf('f_{res} = %.2f GHz', f_res_set1(idx)/1e9));
end
grid on; box on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('透射幅度 (dB)', 'FontSize', 12, 'FontName', 'SimHei');
title('透射幅度随谐振频率的变化', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

% --- Figure 2: 阻尼因子影响（类比论文图3-3b） ---
figure(2); clf;
set(gcf, 'Position', [150, 150, 900, 700]);

% 子图1：群时延
subplot(2,1,1);
hold on;
for idx = 1:3
    plot(f_probe/1e9, tau_group2(idx, :)*1e9, 'LineWidth', 2.5, 'Color', colors(idx, :), ...
        'DisplayName', sprintf('γ = %.2f GHz', gamma_set2(idx)/1e9));
end
xline(f_res_fixed/1e9, 'k--', 'LineWidth', 1.5, 'DisplayName', '谐振频率');
grid on; box on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相对群时延 (ns)', 'FontSize', 12, 'FontName', 'SimHei');
title('固定谐振频率 f_{res} = 35.5 GHz，改变阻尼因子', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

% 子图2：透射幅度
subplot(2,1,2);
hold on;
for idx = 1:3
    plot(f_probe/1e9, amplitude_dB2(idx, :), '--', 'LineWidth', 2.5, 'Color', colors(idx, :), ...
        'DisplayName', sprintf('γ = %.2f GHz', gamma_set2(idx)/1e9));
end
grid on; box on;
xlabel('探测频率 (GHz)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('透射幅度 (dB)', 'FontSize', 12, 'FontName', 'SimHei');
title('透射幅度随阻尼因子的变化', 'FontSize', 14, 'FontName', 'SimHei');
legend('Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

%% 6. 峰值特征分析（谐振频率附近的行为）

fprintf('\n=== 谐振频率附近的峰值特征分析 ===\n');

% 找到谐振峰位置
for idx = 1:3
    [max_tau2, max_idx2] = max(abs(tau_group2(idx, :)));
    f_peak = f_probe(max_idx2);
    
    % 计算半峰宽（FWHM）
    half_max = max_tau2 / 2;
    above_half = abs(tau_group2(idx, :)) > half_max;
    fwhm_points = f_probe(above_half);
    if length(fwhm_points) > 1
        fwhm = (max(fwhm_points) - min(fwhm_points)) / 1e9;  % GHz
    else
        fwhm = NaN;
    end
    
    fprintf('γ = %.2f GHz: 峰值位置 = %.3f GHz, 峰值强度 = %.3f ns, FWHM ≈ %.3f GHz\n', ...
        gamma_set2(idx)/1e9, f_peak/1e9, max_tau2*1e9, fwhm);
end

%% 7. 结果保存

fprintf('\n=== 保存仿真结果 ===\n');

% 创建输出目录
output_dir = fullfile(pwd, '..', 'research_output', '20260115_lorentz_sensitivity');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    mkdir(fullfile(output_dir, 'figures'));
    mkdir(fullfile(output_dir, 'data'));
end

% 保存图表
saveas(figure(1), fullfile(output_dir, 'figures', 'fig1_fres_dominance.png'));
saveas(figure(2), fullfile(output_dir, 'figures', 'fig2_gamma_effect.png'));

% 保存数据
save(fullfile(output_dir, 'data', 'sensitivity_results.mat'), ...
    'f_probe', 'tau_group1', 'tau_group2', 'amplitude_dB1', 'amplitude_dB2', ...
    'f_res_set1', 'gamma_set2', 'S_tau_fres', 'S_tau_gamma');

fprintf('结果已保存至: %s\n', output_dir);

%% 8. 结论总结

fprintf('\n========================================\n');
fprintf('研究结论\n');
fprintf('========================================\n\n');

fprintf('1. 谐振频率敏感度: S_τ^fres = %.2f\n', S_tau_fres);
fprintf('   - 谐振频率变化±1.4%%时，群时延变化%.1f%%\n', delta_tau1_percent*100);
fprintf('   - 在谐振峰附近，时延曲线形态发生显著变化\n\n');

fprintf('2. 阻尼因子敏感度: S_τ^γ = %.2f\n', S_tau_gamma);
fprintf('   - 阻尼因子变化180%%时，群时延变化%.1f%%\n', delta_tau2_percent*100);
fprintf('   - 阻尼主要影响峰值宽度和高度\n\n');

fprintf('3. 敏感度比值: %.2f\n', S_tau_fres / S_tau_gamma);

if abs(S_tau_fres) > 10 * abs(S_tau_gamma)
    fprintf('\n【结论】Lorentz模型中，谐振频率对群时延的影响显著大于阻尼因子。\n');
    fprintf('但与Drude模型不同，阻尼因子的影响并非严格的"二阶小量"，\n');
    fprintf('特别是在谐振频率附近，γ通过改变峰值形态对时延曲线有明显影响。\n');
    fprintf('因此，Lorentz模型的参数反演策略需要区别对待：\n');
    fprintf('  - 若探测频段远离谐振频率，可考虑固定γ仅反演f_res\n');
    fprintf('  - 若探测频段覆盖谐振峰，建议双参数联合反演\n');
else
    fprintf('\n【结论】两个参数对群时延的影响相当，必须进行双参数反演。\n');
end

fprintf('\n仿真完成！\n');
