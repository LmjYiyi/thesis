%% 多解性论证：仅显示可传播频段内的真实曲线交点
clc; clear; close all;

%% ================== 1. 物理常数 ==================
e_charge = 1.602e-19;
me = 9.109e-31;
eps0 = 8.854e-12;
c = 3e8;

nu_val = 1.5e9;                 % 碰撞频率
f_probe = linspace(34e9, 37.5e9, 600);  % 频率轴（点数要足够密）

%% ================== 2. 参数空间 ==================
ne_list = linspace(1.4e19, 1.5e19, 10);
d_list  = linspace(0.20, 0.30, 20);

num_curves = length(ne_list) * length(d_list);
Y_curves = zeros(num_curves, length(f_probe));

% params_map: [ne, d, fp]
params_map = zeros(num_curves, 3);

curve_idx = 0;
for ne = ne_list
    for d = d_list
        curve_idx = curve_idx + 1;

        % 截止频率
        fp = sqrt(ne * e_charge^2 / (eps0 * me)) / (2*pi);

        params_map(curve_idx,:) = [ne, d, fp];

        Y_curves(curve_idx,:) = calculate_theoretical_delay( ...
            f_probe, ne, nu_val, d, c, eps0, me, e_charge);
    end
end

%% ================== 3. 绘制背景曲线 ==================
figure('Color','w','Position',[200 180 950 600]);
hold on; grid on;

plot(f_probe/1e9, Y_curves*1e9, ...
    'Color',[0 0 0 0.12], 'LineWidth',1);

xlabel('探测频率 (GHz)');
ylabel('相对群时延 (ns)');
title('【多解性论证】可传播频段内的真实曲线交点');

%% ================== 4. 精确计算并筛选交点 ==================
fprintf('正在计算可传播频段内的交点...\n');
plot_cnt = 0;

for i = 1:num_curves-1
    for j = i+1:num_curves

        y1 = Y_curves(i,:);
        y2 = Y_curves(j,:);
        diff_y = y1 - y2;

        idx_cross = find(diff_y(1:end-1).*diff_y(2:end) < 0);

        for k = 1:length(idx_cross)

            idx = idx_cross(k);

            % --- 线性插值求交点 ---
            r = abs(diff_y(idx)) / ...
                (abs(diff_y(idx)) + abs(diff_y(idx+1)));

            f_cross = f_probe(idx) + r * ...
                (f_probe(idx+1) - f_probe(idx));

            tau1 = y1(idx) + r * (y1(idx+1) - y1(idx));
            tau2 = y2(idx) + r * (y2(idx+1) - y2(idx));
            tau_cross = (tau1 + tau2)/2;

            % --- 物理筛选：截止频率 ---
            fp_cut = max(params_map(i,3), params_map(j,3));
            if f_cross <= fp_cut
                continue;
            end

            % --- 参数差异筛选（显著多解） ---
            p1 = params_map(i,1:2);
            p2 = params_map(j,1:2);

            dn = abs(p1(1)-p2(1))/mean([p1(1),p2(1)]);
            dd = abs(p1(2)-p2(2))/mean([p1(2),p2(2)]);

            if dn > 0.05 && dd > 0.05
                plot(f_cross/1e9, tau_cross*1e9, ...
                    'ro','MarkerFaceColor','r','MarkerSize',6);
                plot_cnt = plot_cnt + 1;
            end
        end
    end
end

fprintf('共标记 %d 个物理有效交点。\n', plot_cnt);

xlim([min(f_probe) max(f_probe)]/1e9);

%% ================== 5. Drude 相位求导函数 ==================
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
