%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 图3-5 / 图3-6：带宽与色散系数对频轨展宽的影响
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% 1. Constants and parameters
f0 = 34e9;              % start frequency (Hz)
omega0 = 2 * pi * f0;   % angular frequency (rad/s)
T_m = 1e-3;             % modulation period (s)
tau0 = 1e-9;            % zero-order group delay term (s)

% Example dispersion cases
tau1_cases = [-2, 1] * 1e-12;  % s^2
tau2_cases = [1, 1] * 1e-30;   % s^3

% Bandwidth range
B = linspace(1e9, 5e9, 300);   % 1-5 GHz

%% 2. Figure 3-5: broadening vs bandwidth
Kp = 2 * pi * B / T_m;

% Case 1
tau1 = tau1_cases(1);
tau2 = tau2_cases(1);
C1 = omega0 * tau2 + 2 * tau1;
C2 = tau1^2 + tau0 * tau2;
Delta_f_case1 = (2 * pi * B.^2 / T_m) .* abs(C1 - (2 * pi * B / T_m) .* C2);
Delta_f_quad = (2 * pi * B.^2 / T_m) .* abs(C1);
B_zero_case1 = (C1 / C2) * T_m / (2 * pi);

% Case 2
tau1 = tau1_cases(2);
tau2 = tau2_cases(2);
C1 = omega0 * tau2 + 2 * tau1;
C2 = tau1^2 + tau0 * tau2;
Delta_f_case2 = (2 * pi * B.^2 / T_m) .* abs(C1 - (2 * pi * B / T_m) .* C2);
B_zero_case2 = (C1 / C2) * T_m / (2 * pi);

fig1 = figure('Position', [120, 120, 820, 520], 'Color', 'w');

colors = [
    0.0000, 0.4470, 0.7410;
    0.8500, 0.3250, 0.0980;
    0.0000, 0.0000, 0.0000
];

plot(B / 1e9, Delta_f_case1 / 1e6, 'o-', ...
    'Color', colors(1, :), 'LineWidth', 1.8, 'MarkerSize', 4);
hold on;
plot(B / 1e9, Delta_f_case2 / 1e6, 's-', ...
    'Color', colors(2, :), 'LineWidth', 1.8, 'MarkerSize', 4);
plot(B / 1e9, Delta_f_quad / 1e6, '--', ...
    'Color', colors(3, :), 'LineWidth', 2.2);

grid on;
xlabel('带宽 \fontname{Times New Roman}B (GHz)', 'Interpreter', 'tex');
ylabel('频轨展宽 \fontname{Times New Roman}\Delta f_D (MHz)', 'Interpreter', 'tex');
title('频轨展宽随带宽变化');
legend( ...
    '\Delta f_D - B (\tau_1=-2 ps/Hz, \tau_2=1 fs^2/Hz)', ...
    '\Delta f_D - B (\tau_1=1 ps/Hz, \tau_2=1 fs^2/Hz)', ...
    '二次近似 (\propto B^2)', ...
    'Location', 'northwest');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

if isfinite(B_zero_case1) && B_zero_case1 > min(B) && B_zero_case1 < max(B)
    xline(B_zero_case1 / 1e9, ':', 'Color', colors(1, :), 'LineWidth', 1.4);
    text(B_zero_case1 / 1e9 + 0.05, max(Delta_f_case1 / 1e6) * 0.20, ...
        sprintf('B_0^{(1)} = %.2f GHz', B_zero_case1 / 1e9), ...
        'FontSize', 11, 'Color', colors(1, :), 'FontName', 'Times New Roman');
end
if isfinite(B_zero_case2) && B_zero_case2 > min(B) && B_zero_case2 < max(B)
    xline(B_zero_case2 / 1e9, ':', 'Color', colors(2, :), 'LineWidth', 1.4);
    text(B_zero_case2 / 1e9 + 0.05, max(Delta_f_case1 / 1e6) * 0.12, ...
        sprintf('B_0^{(2)} = %.2f GHz', B_zero_case2 / 1e9), ...
        'FontSize', 11, 'Color', colors(2, :), 'FontName', 'Times New Roman');
end

%% 3. Figure 3-6: broadening vs dispersion coefficients
B_fixed = 4e9;
Kp_fixed = 2 * pi * B_fixed / T_m;

tau1_range = linspace(-5, 5, 400) * 1e-12;
tau2_fixed = 1e-30;
C1_tau1 = omega0 * tau2_fixed + 2 * tau1_range;
C2_tau1 = tau1_range.^2 + tau0 * tau2_fixed;
Delta_f_tau1 = (2 * pi * B_fixed^2 / T_m) .* abs(C1_tau1 - Kp_fixed * C2_tau1);

tau2_range = linspace(-5, 5, 400) * 1e-30;
tau1_fixed = 1e-12;
C1_tau2 = omega0 * tau2_range + 2 * tau1_fixed;
C2_tau2 = tau1_fixed^2 + tau0 * tau2_range;
Delta_f_tau2 = (2 * pi * B_fixed^2 / T_m) .* abs(C1_tau2 - Kp_fixed * C2_tau2);

fig2 = figure('Position', [980, 120, 820, 520], 'Color', 'w');
plot(tau1_range / 1e-12, Delta_f_tau1 / 1e6, 'LineWidth', 2.0, 'Color', colors(1, :));
hold on;
plot(tau2_range / 1e-30, Delta_f_tau2 / 1e6, 'LineWidth', 2.0, 'Color', colors(2, :));
grid on;
xlabel('\fontname{Times New Roman}\tau_1 (ps/Hz) \fontname{SimSun}或 \fontname{Times New Roman}\tau_2 (fs^2/Hz)', ...
    'Interpreter', 'tex');
ylabel('频轨展宽 \fontname{Times New Roman}\Delta f_D (MHz)', 'Interpreter', 'tex');
title('频轨展宽随色散系数变化 (B = 4 GHz)');
legend('\Delta f_D - \tau_1 (\tau_2 = 1 fs^2/Hz)', ...
       '\Delta f_D - \tau_2 (\tau_1 = 1 ps/Hz)', ...
       'Location', 'northeast');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on');

%% 4. Export (explicit figure handles)
export_thesis_figure(fig1, '图3-5_带宽色散耦合曲线', 14, 600, 'SimSun');
export_thesis_figure(fig2, '图3-6_色散系数影响频轨展宽', 14, 600, 'SimSun');

fprintf('导出完成：图3-5 与 图3-6。\n');
