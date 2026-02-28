%% 图3-8: 多解性论证 - 不同参数组合下的曲线交点
% 数据来源: MATLAB理论计算 (基于thesis-code/untitled.m)
% 对应章节: 3.2.3 多解性问题
% 注意: 此图保留文字标注用于说明多解性物理含义

clc; clear; close all;

%% 1. 物理常数
e_charge = 1.602e-19;
me = 9.109e-31;
eps0 = 8.854e-12;
c = 3e8;

nu_val = 1.5e9;  % 碰撞频率
f_probe = linspace(34e9, 37.5e9, 600);  % 探测频段

%% 2. 参数空间
ne_list = linspace(1.4e19, 1.5e19, 10);  % 电子密度范围
d_list = linspace(0.20, 0.30, 20);        % 厚度范围

num_curves = length(ne_list) * length(d_list);
Y_curves = zeros(num_curves, length(f_probe));
params_map = zeros(num_curves, 3);  % [ne, d, fp]

curve_idx = 0;
for ne = ne_list
    for d = d_list
        curve_idx = curve_idx + 1;
        
        % 截止频率
        fp = sqrt(ne * e_charge^2 / (eps0 * me)) / (2*pi);
        params_map(curve_idx,:) = [ne, d, fp];
        
        Y_curves(curve_idx,:) = calculate_theoretical_delay(...
            f_probe, ne, nu_val, d, c, eps0, me, e_charge);
    end
end

%% 3. 绘图
figure('Color', 'w', 'Position', [150 100 900 600]);
hold on; grid on;

% 绘制所有曲线 (灰色背景)
plot(f_probe/1e9, Y_curves*1e9, 'Color', [0 0 0 0.12], 'LineWidth', 1);

%% 4. 计算并标记交点
fprintf('正在计算交点...\n');
cross_count = 0;

for i = 1:num_curves-1
    for j = i+1:num_curves
        y1 = Y_curves(i,:);
        y2 = Y_curves(j,:);
        diff_y = y1 - y2;
        
        % 查找符号变化点
        idx_cross = find(diff_y(1:end-1).*diff_y(2:end) < 0);
        
        for k = 1:length(idx_cross)
            idx = idx_cross(k);
            
            % 线性插值求交点
            r = abs(diff_y(idx)) / (abs(diff_y(idx)) + abs(diff_y(idx+1)));
            f_cross = f_probe(idx) + r * (f_probe(idx+1) - f_probe(idx));
            tau1 = y1(idx) + r * (y1(idx+1) - y1(idx));
            tau2 = y2(idx) + r * (y2(idx+1) - y2(idx));
            tau_cross = (tau1 + tau2) / 2;
            
            % 物理筛选: 交点必须在两曲线截止频率之上
            fp_max = max(params_map(i,3), params_map(j,3));
            if f_cross <= fp_max
                continue;
            end
            
            % 参数差异筛选: 排除近似相等的参数组合
            p1 = params_map(i, 1:2);
            p2 = params_map(j, 1:2);
            dn = abs(p1(1)-p2(1)) / mean([p1(1), p2(1)]);
            dd = abs(p1(2)-p2(2)) / mean([p1(2), p2(2)]);
            
            if dn > 0.05 && dd > 0.05
                plot(f_cross/1e9, tau_cross*1e9, 'ro', ...
                    'MarkerFaceColor', 'r', 'MarkerSize', 5);
                cross_count = cross_count + 1;
            end
        end
    end
end

fprintf('共标记 %d 个物理有效交点\n', cross_count);

%% 5. 图形美化
xlabel('探测频率 f (GHz)', 'FontSize', 12);
ylabel('相对群时延 \tau_g (ns)', 'FontSize', 12);
title('多解性论证: 不同(n_e, d)参数组合的群时延曲线交点', 'FontSize', 13);

xlim([34 37.5]);
ylim([-1 9]);

% 标注说明 (此图保留文字标注)
text(35, 7.5, {'灰色曲线: 200组(n_e, d)组合', ...
    sprintf('红色圆点: %d个物理有效交点', cross_count)}, ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5]);

text(36.5, 1.5, {'交点处:', '单频测量无法区分', '不同参数组合'}, ...
    'FontSize', 10, 'Color', [0.6 0 0], 'HorizontalAlignment', 'center');

% 参数范围标注
text(34.2, -0.3, sprintf('n_e \\in [1.4, 1.5]×10^{19} m^{-3}, d \\in [0.20, 0.30] m'), ...
    'FontSize', 9, 'Color', [0.4 0.4 0.4]);

set(gca, 'FontSize', 11);

%% 辅助函数: Drude模型群时延计算
function tau_rel = calculate_theoretical_delay(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
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
