# 项目完整上下文文档

> 本文档为硕士论文《基于滑动时频特征提取与加权非线性反演的色散等离子体诊断研究》提供全面的背景信息和技术细节，用于在新会话中快速恢复上下文。

---

## 一、研究背景与核心问题

### 1.1 研究课题

**论文题目：** 《基于滑动时频特征提取与加权非线性反演的色散等离子体诊断研究》

**学科方向：** 仪器科学与技术（硕士）

**核心问题：** 解决 LFMCW（线性调频连续波）雷达在临近空间高超声速飞行器等离子体鞘套（强色散介质）环境下的**测距失效**和**电子密度反演**问题。

### 1.2 物理背景

当电磁波穿过等离子体时，等离子体作为**频率色散介质**，不同频率的电磁波群速度不同，导致：
- **时变时延**：群时延不再是常数，随频率剧烈变化
- **频谱展宽**（散焦）：差频信号的 FFT 频谱主瓣展宽
- **频率偏移**：一阶色散引起系统性频移误差
- **传统测距公式失效**：$f_{beat} = K \cdot \tau$ 不再适用

### 1.3 核心创新点

1. **滑动窗口时频解耦**：打破全频段 FFT 的"参数依赖死循环"
2. **MDL + ESPRIT 超分辨特征提取**：在短时窗口内精确估计瞬时差频频率
3. **MCMC 贝叶斯反演**：利用 Metropolis-Hastings 算法对后验分布采样，量化不确定性
4. **参数降维策略**：证明碰撞频率 $\nu_e$ 是二阶微扰，可安全固定

---

## 二、核心理论体系

### 2.1 等离子体 Drude 模型

复介电常数：
$$\tilde{\varepsilon}_r(\omega) = \left(1 - \frac{\omega_p^2}{\omega^2 + \nu_e^2}\right) - j\left(\frac{\nu_e}{\omega}\frac{\omega_p^2}{\omega^2 + \nu_e^2}\right)$$

其中 $\omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}}$ 为等离子体特征角频率。

群速度：
$$v_g(\omega) = c\sqrt{1 - \left(\frac{\omega_p}{\omega}\right)^2}$$

物理群时延：
$$\tau_g(f) = \frac{d}{c} \cdot \frac{1}{\sqrt{1 - (f_p/f)^2}}$$

相对群时延（观测量）：
$$\Delta\tau_g(f) = \frac{d}{c}\left(\frac{1}{\sqrt{1 - (f_p/f)^2}} - 1\right), \quad f > f_p$$

### 2.2 LFMCW 信号模型

发射信号：
$$s_{TX}(t) = \exp\left\{j\left(\omega_0 t + \pi\frac{B}{T_m}t^2\right)\right\}$$

色散介质中的二阶时变群时延模型：
$$\tau_g(t) = A_0 + A_1 t + A_2 t^2$$

- $A_0 = \tau_g(\omega_0)$：基础群时延
- $A_1 = \mu \cdot \tau_1(\omega_0)$：一阶色散引起的线性漂移（$\propto B$）
- $A_2 = \frac{1}{2}\mu^2 \cdot \tau_2(\omega_0)$：二阶色散引起的频谱散焦（$\propto B^2$）

差频信号相位：
$$\Delta\phi(t) = \phi_0 + 2\pi f_D' t + \pi\alpha t^2$$

### 2.3 关键参数与判据

**群时延非线性度因子 $\eta$**（分母必须是 1.5 次方）：
$$\eta(f) = \frac{B}{f} \cdot \frac{(f_p/f)^2}{\left[1 - (f_p/f)^2\right]^{3/2}}$$

**工程适用性判据**（"不可能三角"）：
$$B \cdot \eta \cdot \tau_0 \le 1$$

物理意义：带宽 $B$、非线性度 $\eta$、传播距离（$\tau_0 = d/c$）三者的乘积必须小于 1，否则传统测距方法失效。

### 2.4 碰撞频率的二阶微扰特性

通过泰勒展开证明，引入无量纲小量 $\delta = (\nu_e/\omega)^2$：

$$\tau_g(\omega) \approx \frac{d}{c\sqrt{1-(\omega_p/\omega)^2}}\left[1 - \left(\frac{1}{2}\frac{(\omega_p/\omega)^2}{1-(\omega_p/\omega)^2} + \frac{\omega_p^2}{\omega^2}\right)\delta\right]$$

结论：
- **电子密度 $n_e$**：主导项（一阶），决定群时延曲线拓扑
- **碰撞频率 $\nu_e$**：二阶微扰（$(\nu_e/\omega)^2$），对群时延几乎不可观测
- **反演策略**：固定 $\nu_e$ 为先验常数（~1.5 GHz），仅反演 $n_e$

### 2.5 电子密度估算公式

$$n_e \approx \frac{8\pi^2 \varepsilon_0 m_e c}{e^2} \frac{f^2}{d} \tau$$

---

## 三、算法流水线

### 3.1 技术路线

```
LFMCW差频信号 → 滑动窗口切片 → MDL信源估计 → TLS-ESPRIT频率提取
     ↓
频率-时延特征点集 {(f_probe,i, τ_meas,i, A_i)}
     ↓
加权MCMC贝叶斯反演 → 后验分布 → 电子密度 n_e
```

### 3.2 滑动窗口时频解耦

核心逻辑：**以时间定频率**，打破"参数依赖死循环"

- 第 $i$ 个窗口中心时刻 $t_i$ 对应的探测频率：$X_i = f_{start} + K \cdot t_i$（已知量）
- 窗口内差频信号近似单频，ESPRIT 提取 $f_{beat,i}$
- 测量时延：$Y_i = f_{beat,i} / K$

### 3.3 MDL 多径抑制

$$\text{MDL}(k) = -\ln\left(\frac{\prod_{j=k+1}^{M}\lambda_j^{\frac{1}{M-k}}}{\frac{1}{M-k}\sum_{j=k+1}^{M}\lambda_j}\right)^{(M-k)L} + \frac{1}{2}k(2M-k)\ln L$$

### 3.4 ESPRIT 超分辨频率估计

基于旋转不变子空间，不受 FFT 栅栏效应限制，可精确估计非整数倍频率。
取最小频率分量作为直达波的差频（反射路径更长→频率更高）。

### 3.5 MCMC Metropolis-Hastings 反演

贝叶斯后验：$P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$

加权似然函数：
$$\ln L(\theta) = -\frac{1}{2\sigma_0^2}\sum_{i=1}^{N} w_i\left(\tau_{meas,i} - \tau_{model,i}\right)^2$$

幅度权重：$w_i = (A_i / \max(\mathbf{A}))^2$

MH 接受概率：$\alpha = \min\left(1, \exp[\ln L(\theta') - \ln L(\theta^{(t)})]\right)$

参数可观测性判据（CV）：
- CV < 5%：强可观测
- 5-15%：中等
- 15-30%：弱可观测
- ≥ 30%：不可观测

---

## 四、论文大纲与章节状态

### 4.1 完整大纲

```
第一章 绪论
  1.1 研究背景及意义
  1.2 国内外研究现状
  1.3 本文主要研究内容与章节安排

第二章 等离子体电磁特性与LFMCW诊断机理
  2.1 等离子体与电磁波相互作用基础
  2.2 LFMCW时延法诊断原理
  2.3 诊断系统硬件架构与信号链路

第三章 宽带信号在色散介质中的传播机理与方法失效分析 ✅已定稿
  3.1 色散信道的物理建模与参数定义 ✅
  3.2 色散效应下群时延曲线的非线性演化特征
  3.3 色散效应对差频信号的调制机理与误差解析 ✅
  3.4 传统全频段分析方法的适用性边界 ✅

第四章 基于滑动时频特征的贝叶斯反演算法与等离子体验证 ✅已定稿
  4.1 诊断问题的物理约束与降维动机 ✅
  4.2 强色散信号的高分辨率特征提取 ✅
  4.3 基于Metropolis-Hastings的贝叶斯参数反演模型 ✅
  4.4 Drude等离子体模型仿真验证与不确定性量化 ✅

第五章 多物理模型拓展验证与数值研究 ✅已定稿
  5.1 基于CST的Lorentz超材料模型验证 ✅
  5.2 基于Butterworth滤波器的线性系统频率响应参数反演验证 ✅
  5.3 实验系统设计与初步验证 ← 当前工作重点

第六章 总结与展望
```

### 4.2 各章核心发现

| 章节 | 模型 | 主参数（CV） | 次参数（CV） | 关键结论 |
|------|------|-------------|-------------|----------|
| 4.4 | Drude 等离子体 | $n_e$: 0.62% | $\nu_e$: 23.6% | 传统FFT误差245%, MCMC误差<1% |
| 5.1 | Lorentz 超材料 | $f_{res}$: 0.14% | $\gamma$: 48.6% | 跨模型验证参数敏感度层级 |
| 5.2 | Butterworth 滤波器 | $F_0$: 0.29% | $N$: 9.90% | 无空间尺度参数的纯电路系统 |

**普适性结论：** 决定群时延曲线"形状/拓扑"的参数强可观测（CV < 2%），控制"幅度衰减/损耗"的参数弱可观测（CV > 20%）。

---

## 五、代码架构与文件映射

### 5.1 核心 MATLAB 仿真代码 (`thesis-code/`)

| 文件 | 章节 | 功能 |
|------|------|------|
| `LM_MCMC.m` | 4.4 | **核心代码**：Drude等离子体 LFMCW + ESPRIT + MCMC反演 (759行) |
| `LM.m` | 4.4 | Drude LM确定性反演版本 |
| `LFMCW_filter_MCMC.m` | 5.2 | Butterworth滤波器三参数MCMC反演 (810行) |
| `LFMCW_filter_inversion_FINAL.m` | 5.2 | Butterworth滤波器LM反演版本 (603行) |
| `LM_lorentz_MCMC.m` | 5.1 | Lorentz超材料MCMC反演 |
| `LM_lorentz.m` | 5.1 | Lorentz LM反演 |
| `Fig4_9_FFT_vs_ESPRIT_comparison.m` | 4.4 | FFT vs ESPRIT 对比图 (629行) |
| `nue.m` | 4.1 | 碰撞频率敏感性分析 |
| `test.m` | 3.2 | 多解性问题（时延曲线相交） |
| `lianghua.m` | 3.4 | 工程判据 $B \cdot \eta \cdot \tau_0 \le 1$ 量化 |
| `initial.m` | 5.3 | 系统标定与非色散验证 |
| `delay_vs_cutoff_freq.m` | 3.2 | 群时延随截止频率变化可视化 |
| `lorentz_sensitivity_analysis.m` | 5.1 | Lorentz模型参数敏感性 |
| `filter_inversion_feasibility.m` | 5.2 | 滤波器反演可行性分析 |

### 5.2 CST 仿真相关 (`thesis-code/cst_lorentz/`, `thesis-code/cst_filter/`)

- `cst_lorentz/`：CSRR加载波导 CST 仿真后处理代码
- `cst_filter/`：CST 滤波器仿真代码

### 5.3 ADS 仿真相关 (`simulation/ADS/`)

| 文件 | 功能 |
|------|------|
| `plot_all_results.m` | 可视化所有 ADS 仿真结果（频谱/时域/S参数/延迟） |
| `smooth_ADS_data.m` | 发射信号频谱平滑与 -3dB 带宽验证 |
| `process_hunpin_esprit.m` | 从 ADS 混频信号提取时延-频率散点（ESPRIT） |
| `verify_lfmcw_linear.m` | LFMCW 线性度验证（STFT + Hilbert变换） |

ADS 数据文件：
- `s21.txt` / `delay.txt`：S21幅值(dB) 和 群延迟(s)，频率从 ~30 GHz 开始
- `fashe_*.txt`：发射信号（时域 + 频谱）
- `jieshou_*.txt`：接收信号
- `hunpin_*.txt`：混频信号

### 5.4 CST 仿真数据 (`simulation/cst_data/`)

- `output.txt` / `output_air.txt`：CST 时域仿真导出（超材料 / 空气参考）
- `CSRR_WR28_Lorentz.s2p`：Touchstone S 参数文件
- `low_density.txt` / `high_density.txt`：CST 等离子体层 S21 仿真

### 5.5 仿真结果日志 (`simulation/matlab_data/`)

- `initial.txt`：简单单频方法误差（20-33 GHz 截止频率），误差从 1.5% 到 245%
- `MCMC.txt`：MCMC 反演结果（电子密度扫描）
- `SNR.txt`：SNR 敏感性（10-30 dB）
- `nue.txt`：碰撞频率敏感性
- `参数扫描.txt`：联合参数扫描

### 5.6 定稿章节 (`final_output/`)

所有已通过评审的最终版章节：
- `第3章_3.1_色散信道的物理建模与参数定义_final.md`
- `第3章_3.3_色散效应对差频信号的调制机理与误差解析_final.md`
- `第3章_3.4_传统全频段分析方法的适用性边界_final.md`
- `第4章_4.1~4.4_*.md`
- `第5章_5.1~5.2_*.md`

配图代码在 `final_output/figure_code/`（24个MATLAB脚本），生成图片在 `final_output/figures/`（81个文件）。

---

## 六、当前研究重点：带通滤波器作为等效色散介质

### 6.1 研究动机

论文第 5.3 节"实验系统设计与初步验证"需要一个**实验可行的色散介质等效方案**。真实等离子体实验条件苛刻，因此提出使用**带通滤波器**模拟色散介质的群时延特性。

带通滤波器作为等效色散介质的核心逻辑：
- 带通滤波器在通带边沿附近具有**非线性群时延特性**
- 这种群时延随频率的变化模式在数学上类似于等离子体的色散效应
- 可以利用滤波器验证 LFMCW 诊断算法的普适性

### 6.2 实验滤波器参数 (`resources/parameters.txt`)

```
中心频率(F0): 14 GHz
通带频率: 10 ~ 18 GHz
通带插入损耗: ≤2.0 dB
通带纹波: ≤1.2 dB
通带回波损耗: ≥17 dB
阻带抑制: ≥80 dB @ DC~7.7 GHz, ≥50 dB @ 7.7~8.8 GHz
          ≥50 dB @ 18.5~18.8 GHz, ≥80 dB @ 18.8~25.5 GHz
阻抗: 50 Ohms
功率: 5W Max
连接器: SMA-Female
尺寸: 90*20*11mm
材料: H62 铜合金
```

### 6.3 Butterworth 滤波器仿真模型

**已完成验证**（第5.2节定稿）。仿真中使用的 Butterworth 模型：

幅度响应：
$$|H_{BP}(f)|^2 = \frac{1}{1 + \left(\frac{f - F_0}{BW/2}\right)^{2N}}$$

群时延（解析式）：
$$\tau_g = \frac{2N}{\pi \cdot BW}\left[1 + \left(\frac{f-F_0}{BW/2}\right)^2\right]^{-(N+1)/2}$$

仿真参数：
- 频率范围：10-18 GHz
- 真值：$F_0 = 14$ GHz, $BW = 8$ GHz, $N = 5$
- 参考时延：$\tau_{ref} = 2$ ns
- 调制周期：$T_m = 100\ \mu s$

MCMC 反演结果：
- $F_0$ 误差 0.14%（CV=0.29%）
- $BW$ 误差 1.00%（CV=4.42%）
- $N$ 误差 3.00%（CV=9.90%）

### 6.4 ADS 仿真现状

**当前正在进行的工作**：使用 ADS（Advanced Design System）对带通滤波器进行电路级仿真，获取真实的 S 参数和群延迟数据。

已有的 ADS 数据：
- **`s21.txt`**：S21 幅值（dB），频率范围 ~30-40 GHz，10002 个数据点
  - 格式：`freq\tdB(S(2,1))`
  - 当前数据在 30 GHz 处 S21 约 -194 dB（**非常大的衰减**）
  
- **`delay.txt`**：群延迟，频率范围 ~30-40 GHz，10004 个数据点
  - 格式：`freq\tdelay(2,1)`
  - 延迟值约 1.88 ps 量级

- 发射/接收/混频信号时域和频谱数据完整

**关键 ADS 处理脚本**：
- `process_hunpin_esprit.m`：从混频信号提取时延（LFMCW参数：34.2-37.4 GHz, B=3.2 GHz）
- `verify_lfmcw_linear.m`：验证 VCO 产生的 LFMCW 线性度
- `smooth_ADS_data.m`：频谱平滑与带宽验证

### 6.5 三层仿真验证架构

```
Level 1: MATLAB 数值仿真（理想模型）
  ├── Drude 等离子体 (LM_MCMC.m)
  ├── Lorentz 超材料 (LM_lorentz_MCMC.m)
  └── Butterworth 滤波器 (LFMCW_filter_MCMC.m)

Level 2: ADS 电路仿真（真实器件模型）
  ├── LFMCW VCO + 混频器 + 滤波器链路
  └── S参数提取与群延迟验证

Level 3: CST 全波电磁仿真（最高保真度）
  ├── CSRR 加载 WR-28 波导 (Lorentz)
  └── 等离子体层 S21 (Drude)
```

---

## 七、核心算法代码摘要

### 7.1 Drude 群时延计算函数

```matlab
function [delay_relative, freq_out] = calculate_theoretical_delay(n_e, nu_e, d, freq_range)
    c = 3e8;
    epsilon_0 = 8.854e-12;
    m_e = 9.109e-31;
    q_e = 1.602e-19;
    
    omega_p = sqrt(n_e * q_e^2 / (epsilon_0 * m_e));
    omega = 2 * pi * freq_range;
    
    % 复介电常数 (Drude)
    epsilon_r = 1 - omega_p^2 ./ (omega.^2 + nu_e^2) + ...
                1j * (nu_e * omega_p^2) ./ (omega .* (omega.^2 + nu_e^2));
    
    % 复波数
    k = omega / c .* sqrt(epsilon_r);
    
    % 相位与群时延
    phi = -real(k) * d;
    tau_g = -diff(phi) ./ diff(omega);  % 数值微分
    
    tau_vacuum = d / c;
    delay_relative = tau_g - tau_vacuum;
    freq_out = freq_range(1:end-1);
end
```

### 7.2 Butterworth 群时延解析式

```matlab
function tau_g = calculate_filter_group_delay(f, F0, BW, N)
    x = (f - F0) / (BW/2);
    tau_g = (2*N) / (pi * BW) * (1 + x.^2).^(-(N+1)/2);
end
```

### 7.3 MCMC 核心循环（简化版）

```matlab
% 初始化
theta = theta_init;  % [n_e, nu_e] 或 [F0, BW, N]
logL = compute_log_likelihood(theta, data, weights);

for i = 1:N_samples
    % 随机游走提议
    theta_prop = theta + randn(size(theta)) .* proposal_std;
    
    % 先验约束检查
    if all(theta_prop >= prior_min & theta_prop <= prior_max)
        logL_prop = compute_log_likelihood(theta_prop, data, weights);
        
        % MH 接受准则
        if log(rand) < (logL_prop - logL)
            theta = theta_prop;
            logL = logL_prop;
        end
    end
    
    chain(i,:) = theta;
end

% 丢弃 burn-in，计算后验统计
posterior = chain(burn_in+1:end, :);
theta_mean = mean(posterior);
theta_std = std(posterior);
CV = theta_std ./ abs(theta_mean) * 100;  % 变异系数
```

### 7.4 加权对数似然函数

```matlab
function logL = compute_log_likelihood(theta, f_probe, tau_meas, weights, sigma)
    tau_theory = forward_model(theta, f_probe);  % 正向模型
    residuals = tau_meas - tau_theory;
    logL = -0.5 / sigma^2 * sum(weights .* residuals.^2);
end
```

### 7.5 ESPRIT 特征提取核心

```matlab
% 滑动窗口循环
for i = 1:N_windows
    segment = if_signal(win_start:win_end);
    
    % 构建 Hankel 矩阵
    H = hankel(segment(1:M), segment(M:end));
    
    % 前向-后向空间平滑
    J = fliplr(eye(size(H,2)));
    R = (H*H' + J*(H*H')'*J) / (2*size(H,2));
    
    % 特征分解 + MDL 定阶
    [V, D] = eig(R);
    eigenvalues = sort(diag(D), 'descend');
    k = MDL_criterion(eigenvalues, size(H,2));
    
    % TLS-ESPRIT
    Es = V(:, end-k+1:end);  % 信号子空间
    E1 = Es(1:end-1, :);
    E2 = Es(2:end, :);
    phi = eig(pinv(E1) * E2);
    f_beat = Fs/(2*pi) * angle(phi);
    
    % 选择最小频率（直达波）
    feature_freq(i) = min(abs(f_beat));
end
```

---

## 八、关键仿真参数汇总

### 8.1 Drude 等离子体（第4章）

| 参数 | 符号 | 数值 |
|------|------|------|
| 起始频率 | $f_{start}$ | 34.2 GHz |
| 终止频率 | $f_{end}$ | 37.4 GHz |
| 带宽 | $B$ | 3.2 GHz |
| 调制周期 | $T_m$ | 50 μs |
| 等离子体厚度 | $d$ | 150 mm |
| 截止频率 | $f_p$ | 20-33 GHz (可变) |
| 碰撞频率 | $\nu_e$ | 1.5 GHz (默认) |
| 信噪比 | SNR | 20 dB (默认) |

### 8.2 Lorentz 超材料（第5.1章）

| 参数 | 符号 | 数值 |
|------|------|------|
| 波导类型 | — | WR-28 |
| 等效厚度 | $d$ | 3 mm |
| 等效等离子频率 | $\omega_p$ | $2\pi \times 5$ GHz |
| 谐振频率 | $f_{res}$ | 34.5 GHz |
| 阻尼系数 | $\gamma$ | 0.5 GHz |

### 8.3 Butterworth 滤波器（第5.2章）

| 参数 | 符号 | 数值 |
|------|------|------|
| 中心频率 | $F_0$ | 14 GHz |
| 带宽 | $BW$ | 8 GHz |
| 阶数 | $N$ | 5 |
| LFMCW频率范围 | — | 10-18 GHz |
| 调制周期 | $T_m$ | 100 μs |
| 参考时延 | $\tau_{ref}$ | 2 ns |

### 8.4 ADS 仿真（当前）

| 参数 | 数值 |
|------|------|
| LFMCW 范围 | 34.2-37.4 GHz |
| 带宽 | 3.2 GHz |
| 滤波器中心频率 | 14 GHz |
| 通带 | 10-18 GHz |

### 8.5 物理常数

| 常数 | 符号 | 数值 |
|------|------|------|
| 光速 | $c$ | $3 \times 10^8$ m/s |
| 真空介电常数 | $\varepsilon_0$ | $8.854 \times 10^{-12}$ F/m |
| 电子质量 | $m_e$ | $9.109 \times 10^{-31}$ kg |
| 电子电量 | $e$ | $1.602 \times 10^{-19}$ C |

---

## 九、论文写作规范与约束

### 9.1 不可更改的理论准则

1. **碰撞频率是二阶微扰**：$\nu_e$ 对群时延贡献为 $(\nu_e/\omega)^2$
2. **多解性问题**：不同参数组合的时延曲线可能相交，需曲线拟合
3. **反演策略**：固定 $\nu_e$，仅反演 $n_e$
4. **非线性度因子分母 1.5 次方**
5. **工程判据**：$B \cdot \eta \cdot \tau_0 \le 1$

### 9.2 写作风格

- 禁止使用"我"、"我们"，使用"本文"、"本系统"
- 公式与物理意义融合（"夹叙夹议"）
- 禁止列表式写作，使用"流体感"段落
- 图表融合到正文分析中，禁止独立的 `[可视化建议]`
- 中性学术词汇，禁止"彻底打破"、"完美"等绝对表述

### 9.3 符号统一规范

| 物理量 | 标准符号 | 说明 |
|--------|---------|------|
| 群时延 | $\tau_g$ | |
| 相对群时延 | $\Delta\tau_g$ | |
| 非线性度因子 | $\eta$ | |
| 截止频率 | $f_p$ | |
| 碰撞频率 | $\nu_e$ | |
| 电子密度 | $n_e$ | |
| 调频斜率 | $K = B/T_m$ | 线性频率域 |
| 角频率调频斜率 | $\mu = 2\pi B/T_m$ | 角频率域 |

---

## 十、项目目录结构

```
thesis/
├── resources/                    # 核心参考资源
│   ├── 核心研究档案.txt           # 完整理论推导（~1100行）
│   ├── 大纲.txt                  # 论文6章大纲
│   ├── role.txt                  # AI角色与理论准则
│   ├── parameters.txt            # 实验滤波器参数
│   ├── writing-style.txt         # 写作风格参考
│   ├── formula-templates.md      # LaTeX公式模板
│   ├── matlab-code-reference.md  # 代码与章节映射
│   └── project-context.md        # 本文档
│
├── thesis-code/                  # MATLAB 仿真代码
│   ├── LM_MCMC.m                # 核心：Drude MCMC反演
│   ├── LFMCW_filter_MCMC.m      # Butterworth滤波器MCMC
│   ├── LM_lorentz_MCMC.m        # Lorentz MCMC反演
│   ├── Fig4_9_FFT_vs_ESPRIT_comparison.m
│   ├── nue.m / test.m / lianghua.m / initial.m
│   ├── cst_lorentz/             # CST Lorentz后处理
│   └── cst_filter/              # CST 滤波器后处理
│
├── simulation/                   # 仿真数据与处理
│   ├── ADS/                     # ADS电路仿真
│   │   ├── s21.txt / delay.txt  # S参数与群延迟
│   │   ├── fashe_*.txt          # 发射信号
│   │   ├── jieshou_*.txt        # 接收信号
│   │   ├── hunpin_*.txt         # 混频信号
│   │   ├── plot_all_results.m   # 结果可视化
│   │   ├── process_hunpin_esprit.m  # ESPRIT时延提取
│   │   ├── smooth_ADS_data.m    # 频谱平滑
│   │   └── verify_lfmcw_linear.m    # 线性度验证
│   ├── cst_data/                # CST仿真数据
│   │   ├── output.txt / output_air.txt
│   │   └── CSRR_WR28_Lorentz.s2p
│   ├── matlab_data/             # MATLAB结果日志
│   ├── CST_TimeDomain_MCMC.m    # CST数据MCMC反演
│   └── CST_TimeDomain_Mixing.m  # CST数据基本混频
│
├── final_output/                 # 已定稿论文章节
│   ├── 第3章_*.md / 第4章_*.md / 第5章_*.md
│   ├── figure_code/             # 配图MATLAB脚本(24个)
│   ├── figures/                 # 生成图片(81个)
│   └── sim_data/                # 仿真数据
│
├── output/                      # 各版本草稿
├── reviews/                     # 评审报告
├── research_output/             # 科研探索成果
│
├── thesis-writing.md            # 写作工作流
├── expert-review.md             # 评审工作流
├── research-learning.md         # 科研学习工作流
└── finalize.md                  # 定稿工作流
```

---

## 十一、下一步工作方向

### 11.1 第5.3节：实验系统设计与初步验证

1. **系统标定**：利用 `initial.m` 代码验证非色散环境下的系统精度
2. **色散介质等效**：利用带通滤波器（14 GHz中心，10-18 GHz通带）作为等效色散介质
3. **ADS 仿真验证**：
   - 从 ADS 导出的 `s21.txt` 和 `delay.txt` 验证滤波器群延迟特性
   - 利用 `process_hunpin_esprit.m` 从混频信号提取时延特征
   - 将提取结果与理论 Butterworth 模型对比

### 11.2 ADS 仿真数据处理流程

```
ADS 导出 → s21.txt + delay.txt → 群延迟特性分析
                                      ↓
ADS 导出 → hunpin_time_v.txt → process_hunpin_esprit.m → 时延-频率散点
                                      ↓
                    与 LFMCW_filter_MCMC.m 的理论曲线对比
                                      ↓
                    验证带通滤波器作为等效色散介质的可行性
```

### 11.3 待解决的关键问题

1. ADS s21.txt 当前显示 ~-194 dB 的极大衰减，需检查 ADS 仿真设置是否正确
2. 群延迟数据（~1.88 ps）量级较小，需确认是否为滤波器本身的群延迟还是仿真噪声
3. 需要确保 ADS LFMCW 信号频率范围与滤波器通带匹配（当前 ADS 设置为 34.2-37.4 GHz，但滤波器通带为 10-18 GHz）
