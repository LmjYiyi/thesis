%% 不同截止频率下的相对群时延曲线（Drude 模型，厚度固定 150 mm）
% 仿造 untitled.m 的时延计算方式，绘制多条 fp 从 15 GHz 到 34 GHz 的曲线
% 横纵轴频率范围：30–40 GHz（横轴为探测频率，纵轴为相对群时延）
clc; clear; close all;

%% ================== 1. 物理常数 ==================
e_charge = 1.602e-19;
me = 9.109e-31;
eps0 = 8.854e-12;
c = 3e8;

nu_val = 1.5e9;                    % 碰撞频率 (Hz)
d = 0.15;                          % 厚度固定 150 mm

%% ================== 2. 频率与截止频率设置 ==================
f_probe = linspace(30e9, 40e9, 800);   % 横轴：30–40 GHz
fp_list = linspace(25e9, 33e9, 4);    % 截止频率 15–34 GHz，取 15 条曲线

% 由 fp 反推 ne：fp = sqrt(ne * e^2 / (eps0*me)) / (2*pi) => ne = fp^2 * (2*pi)^2 * eps0 * me / e^2
ne_list = (fp_list * 2*pi).^2 * eps0 * me / (e_charge^2);

num_curves = length(fp_list);
Y_curves = zeros(num_curves, length(f_probe));

for k = 1:num_curves
    ne = ne_list(k);
    Y_curves(k,:) = calculate_theoretical_delay( ...
        f_probe, ne, nu_val, d, c, eps0, me, e_charge);
end

%% ================== 3. 绘图 ==================
figure('Color', 'w', 'Position', [200 180 900 560]);
hold on; grid on;

% 颜色：从低 fp 到高 fp 渐变（如蓝→绿→红）
cmap = parula(num_curves);
for k = 1:num_curves
    plot(f_probe/1e9, Y_curves(k,:)*1e9, ...
        'Color', cmap(k,:), 'LineWidth', 1.5);
end

% 坐标轴与标题（中文用 SimHei）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
xlabel('探测频率 (GHz)', 'FontName', 'SimHei', 'FontSize', 12);
ylabel('相对群时延 (ns)', 'FontName', 'SimHei', 'FontSize', 12);
title('不同截止频率下的相对群时延（Drude，d=150 mm）', 'FontName', 'SimHei', 'FontSize', 12);

xlim([30 40]);
ylim_auto = [min(Y_curves(:))*1e9, max(Y_curves(:))*1e9];
ylim(ylim_auto);

% 添加35GHz和37GHz的垂直虚线
line([35 35], ylim_auto, 'LineStyle', '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
line([37 37], ylim_auto, 'LineStyle', '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);

% 在截止频率30GHz的曲线上，35-37GHz区间内标记几个点
fp_30GHz = 30e9;
% 找到最接近30GHz的截止频率索引
[~, idx_30] = min(abs(fp_list - fp_30GHz));
% 在35-37GHz区间均匀取8个点
f_markers = linspace(35e9, 37e9, 8);
for i = 1:length(f_markers)
    % 找到对应频率的索引
    [~, f_idx] = min(abs(f_probe - f_markers(i)));
    % 在该点处标记
    plot(f_probe(f_idx)/1e9, Y_curves(idx_30, f_idx)*1e9, ...
        'o', 'MarkerSize', 8, 'MarkerFaceColor', cmap(idx_30,:), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
end

% 图例：显示各曲线对应的 fp (GHz)
leg_entries = arrayfun(@(fp) sprintf('f_p = %.0f GHz', fp/1e9), fp_list, 'UniformOutput', false);
legend(leg_entries, 'Location', 'best', 'FontSize', 8, 'NumColumns', 2);

hold off;

%% ================== 4. Drude 相对群时延计算（与 untitled.m 一致） ==================
function tau_rel = calculate_theoretical_delay( ...
        f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)

    omega = 2*pi*f_vec;
    wp = sqrt(ne_val * e_charge^2 / (eps0 * me));

    eps_r = 1 - (wp^2) ./ (omega .* (omega + 1i*nu_val));
    k = (omega./c) .* sqrt(eps_r);

    phi = -real(k) * d;

    dphi = diff(phi);
    domega = diff(omega);

    tau = -dphi ./ domega;
    tau = [tau tau(end)];

    tau_rel = tau - d/c;
end
