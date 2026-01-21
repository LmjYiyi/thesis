%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 从CST S21数据提取Lorentz模型参数
% 功能: 直接拟合S21数据获取f_res和gamma的"真实值"，用于验证MCMC反演
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

%% 1. 读取S参数数据
s2p_file = fullfile(fileparts(mfilename('fullpath')), 'cst_lorentz', 'data', 'data.s2p');
[f_cst, S11, S21, S12, S22] = read_s2p_touchstone(s2p_file);

fprintf('========================================\n');
fprintf('从S21数据提取Lorentz参数\n');
fprintf('========================================\n');

%% 2. 方法1: 从S21幅度曲线直接提取
S21_mag = abs(S21);
S21_dB = 20*log10(S21_mag);

% 谐振频率 = S21最小值点
[S21_min, idx_res] = min(S21_mag);
f_res_direct = f_cst(idx_res);

fprintf('\n--- 方法1: 直接从S21曲线提取 ---\n');
fprintf('谐振频率 f_res = %.4f GHz (S21最小点)\n', f_res_direct/1e9);
fprintf('S21谐振处幅度 = %.2f dB\n', S21_dB(idx_res));

% 3dB带宽估计gamma
S21_3dB_level = S21_min * sqrt(2);  % -3dB点
idx_above_3dB = find(S21_mag > S21_3dB_level);

% 找谐振点左侧和右侧的-3dB点
idx_left = max(idx_above_3dB(idx_above_3dB < idx_res));
idx_right = min(idx_above_3dB(idx_above_3dB > idx_res));

if ~isempty(idx_left) && ~isempty(idx_right)
    BW_3dB = f_cst(idx_right) - f_cst(idx_left);
    gamma_direct = BW_3dB / 2;  % 对于Lorentz谐振，HWHM ≈ gamma
    fprintf('-3dB带宽 = %.4f GHz\n', BW_3dB/1e9);
    fprintf('阻尼因子 gamma ≈ %.4f GHz (HWHM)\n', gamma_direct/1e9);
else
    gamma_direct = 0.5e9;  % 默认值
    fprintf('无法从-3dB带宽估计gamma，使用默认值\n');
end

%% 3. 方法2: 非线性最小二乘拟合S21
fprintf('\n--- 方法2: Lorentz模型直接拟合S21 ---\n');

% 选择拟合频率范围 (谐振附近)
fit_range = (f_cst >= 33e9) & (f_cst <= 36e9);
f_fit = f_cst(fit_range);
S21_fit = S21_mag(fit_range);

% 归一化以便拟合
S21_norm = S21_fit / max(S21_fit);

% Lorentz谐振形状拟合
lorentz_model = @(p, f) 1 ./ sqrt(1 + ((f.^2 - p(1)^2) ./ (p(2)*f)).^2);

% 初始猜测
p0 = [f_res_direct, gamma_direct];

% 拟合选项
opts = optimoptions('lsqcurvefit', 'Display', 'off', 'MaxIterations', 1000);
lb = [33e9, 0.01e9];
ub = [36e9, 5e9];

try
    [p_fit, resnorm] = lsqcurvefit(lorentz_model, p0, f_fit, S21_norm, lb, ub, opts);
    f_res_fit = p_fit(1);
    gamma_fit = p_fit(2);
    
    fprintf('拟合结果:\n');
    fprintf('  f_res = %.4f GHz\n', f_res_fit/1e9);
    fprintf('  gamma = %.4f GHz\n', gamma_fit/1e9);
    fprintf('  残差范数 = %.6f\n', resnorm);
catch ME
    fprintf('拟合失败: %s\n', ME.message);
    f_res_fit = f_res_direct;
    gamma_fit = gamma_direct;
end

%% 4. 综合结果
fprintf('\n========================================\n');
fprintf('综合结果汇总\n');
fprintf('========================================\n');
fprintf('方法          |  f_res (GHz)  |  gamma (GHz)\n');
fprintf('----------------------------------------\n');
fprintf('S21幅度最小   |  %.4f      |  %.4f\n', f_res_direct/1e9, gamma_direct/1e9);
fprintf('Lorentz拟合   |  %.4f      |  %.4f\n', f_res_fit/1e9, gamma_fit/1e9);
fprintf('----------------------------------------\n');

% 最佳估计（取平均）
f_res_best = mean([f_res_direct, f_res_fit]);
gamma_best = mean([gamma_direct, gamma_fit]);

fprintf('\n★ 推荐参数 (用于对比MCMC结果):\n');
fprintf('  f_res = %.4f GHz\n', f_res_best/1e9);
fprintf('  gamma = %.4f GHz\n', gamma_best/1e9);

%% 5. 可视化
figure('Position', [100, 100, 900, 400]);

% S21幅度
subplot(1,2,1);
plot(f_cst/1e9, S21_dB, 'b', 'LineWidth', 1.5);
hold on;
xline(f_res_direct/1e9, 'r--', 'LineWidth', 2);
if ~isempty(idx_left) && ~isempty(idx_right)
    xline(f_cst(idx_left)/1e9, 'g:', 'LineWidth', 1.5);
    xline(f_cst(idx_right)/1e9, 'g:', 'LineWidth', 1.5);
end
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('|S21| (dB)', 'FontName', 'SimHei');
title('S21幅度响应', 'FontName', 'SimHei');
grid on; 
legend('S21', sprintf('f_{res}=%.2f GHz', f_res_direct/1e9), '-3dB边界');

% S21相位
subplot(1,2,2);
phase_cst = unwrap(angle(S21));
plot(f_cst/1e9, rad2deg(phase_cst), 'b', 'LineWidth', 1.5);
hold on;
xline(f_res_direct/1e9, 'r--', 'LineWidth', 2);
xlabel('频率 (GHz)', 'FontName', 'SimHei');
ylabel('∠S21 (°)', 'FontName', 'SimHei');
title('S21相位响应', 'FontName', 'SimHei');
grid on;

sgtitle('CST S21数据分析 - Lorentz参数提取', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'SimHei');

fprintf('\n绘图完成!\n');

%% =========================================================================
%  局部函数
%  =========================================================================

function [f_Hz, S11, S21, S12, S22] = read_s2p_touchstone(filename)
    fid = fopen(filename, 'r');
    if fid == -1, error('无法打开文件: %s', filename); end
    
    freq_unit_factor = 1e9;
    format_type = 'MA';
    data = [];
    
    while ~feof(fid)
        line = fgetl(fid);
        if isempty(line), continue; end
        if line(1) == '!', continue; end
        if line(1) == '#'
            tokens = strsplit(upper(line));
            if any(strcmp(tokens, 'HZ')), freq_unit_factor = 1; end
            if any(strcmp(tokens, 'KHZ')), freq_unit_factor = 1e3; end
            if any(strcmp(tokens, 'MHZ')), freq_unit_factor = 1e6; end
            if any(strcmp(tokens, 'GHZ')), freq_unit_factor = 1e9; end
            if any(strcmp(tokens, 'RI')), format_type = 'RI'; end
            if any(strcmp(tokens, 'MA')), format_type = 'MA'; end
            if any(strcmp(tokens, 'DB')), format_type = 'DB'; end
            continue;
        end
        values = sscanf(line, '%f');
        if length(values) >= 9
            data = [data; values(1:9)'];
        end
    end
    fclose(fid);
    
    f_Hz = data(:, 1) * freq_unit_factor;
    
    if strcmp(format_type, 'MA')
        S11 = data(:, 2) .* exp(1i * data(:, 3) * pi / 180);
        S21 = data(:, 4) .* exp(1i * data(:, 5) * pi / 180);
        S12 = data(:, 6) .* exp(1i * data(:, 7) * pi / 180);
        S22 = data(:, 8) .* exp(1i * data(:, 9) * pi / 180);
    elseif strcmp(format_type, 'RI')
        S11 = data(:, 2) + 1i * data(:, 3);
        S21 = data(:, 4) + 1i * data(:, 5);
        S12 = data(:, 6) + 1i * data(:, 7);
        S22 = data(:, 8) + 1i * data(:, 9);
    elseif strcmp(format_type, 'DB')
        S11 = 10.^(data(:, 2)/20) .* exp(1i * data(:, 3) * pi / 180);
        S21 = 10.^(data(:, 4)/20) .* exp(1i * data(:, 5) * pi / 180);
        S12 = 10.^(data(:, 6)/20) .* exp(1i * data(:, 7) * pi / 180);
        S22 = 10.^(data(:, 8)/20) .* exp(1i * data(:, 9) * pi / 180);
    end
end
