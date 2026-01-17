# 参数敏感性分析与降维策略研究报告

**研究日期**：2026-01-18  
**研究目标**：分析 Lorentz 模型与 Butterworth 滤波器模型的参数对群时延的敏感性，确定何时采用降维策略

---

## 1. 研究背景

在非线性参数反演中，若待反演参数对观测量的敏感性差异悬殊，会导致Jacobian矩阵条件数过大（病态问题）。此时优化算法收敛慢、易陷入局部极值，且低灵敏度参数在噪声下波动剧烈。

**降维策略**的核心思想：固定低灵敏度参数为先验值，仅反演主导参数，将病态的多维优化转化为良态的低维问题。

---

## 2. 理论分析

### 2.1 Lorentz模型群时延

复介电常数：
$$\varepsilon_r(\omega) = 1 + \frac{\omega_p^2}{\omega_0^2 - \omega^2 - j\gamma\omega}$$

相对群时延通过相位对频率求导获得：$\tau_g = -\frac{d\phi}{d\omega} - \frac{d}{c}$

**参数角色**：
- $f_{res}$：控制谐振峰位置（强敏感）
- $\gamma$：控制谐振峰宽度和高度（弱敏感）

### 2.3 Drude模型群时延（等离子体）

复介电常数：
$$\varepsilon_r(\omega) = 1 - \frac{\omega_p^2}{\omega(\omega + j\nu_e)}$$

**参数角色**：
- $f_p$ (或 $n_e$)：决定截止频率，强敏感（尤其在截止频率附近）
- $\nu_e$：阻尼项，引起幅度衰减，对时延为二阶微扰（弱敏感）

### 2.4 Butterworth滤波器群时延

解析公式（来源于相位响应求导）：
$$\tau_g(f) = \frac{2N}{\pi \cdot BW} \cdot \left[1 + \left(\frac{f - F_0}{BW/2}\right)^2\right]^{-(N+1)/2}$$

**参数角色**：
- $F_0$：平移时延曲线（强敏感）
- $BW$：同时影响峰值高度和宽度（中等敏感）
- $N$：控制峰值高度和边缘陡峭度（可能低敏感）

---

## 3. 仿真方案

### 3.1 Lorentz模型参数扫描

| 参数 | 基准值 | 扫描范围 |
|-----|-------|---------|
| $f_{res}$ | 35.5 GHz | 34.5 ~ 36.5 GHz (±3%) |
| $\gamma$ | 0.5 GHz | 0.1 ~ 1.0 GHz (±100%) |

### 3.2 Drude模型参数扫描

| 参数 | 基准值 | 扫描范围 |
|-----|-------|---------|
| $f_p$ | 29 GHz | 28 ~ 30 GHz |
| $\nu_e$ | 1.5 GHz | 0.1 ~ 5.0 GHz |

### 3.3 Butterworth滤波器参数扫描

| 参数 | 基准值 | 扫描范围 |
|-----|-------|---------|
| $F_0$ | 14 GHz | 12 ~ 16 GHz (±15%) |
| $BW$ | 8 GHz | 6 ~ 10 GHz (±25%) |
| $N$ | 5 | 2, 4, 6, 8 |

---

## 4. 敏感性量化方法

### 4.1 归一化敏感性

$$S_\theta = \frac{\partial \tau_g / \tau_g}{\partial \theta / \theta} = \frac{\partial \tau_g}{\partial \theta} \cdot \frac{\theta}{\tau_g}$$

物理意义：参数相对变化1%时，时延的相对变化百分比。

### 4.2 Jacobian条件数

对于多参数优化问题，构建Jacobian矩阵：
$$\mathbf{J}_{ij} = \frac{\partial \tau_g(f_i)}{\partial \theta_j}$$

条件数 $\kappa(\mathbf{J}) = \sigma_{max} / \sigma_{min}$ 反映病态程度：
- $\kappa < 10$：良态，可同时反演
- $\kappa > 100$：病态，建议降维

---

## 5. 降维策略决策准则

### 准则1：敏感性比值准则
若 $|S_{\theta_1}| / |S_{\theta_2}| > 5$，则 $\theta_2$ 应固定为先验值。

### 准则2：条件数准则
若 $\kappa(\mathbf{J}) > 100$，应移除敏感性最低的参数。

### 准则3：物理先验准则
若某参数可通过独立手段获得（如材料规格书、光谱测量），优先固定该参数。

---

## 6. 代码实现

完整仿真代码位于：
- `code/sensitivity_analysis_lorentz_filter.m`

代码运行后将直接显示以下图表：
- Figure 1: Lorentz参数敏感性
- Figure 2: Butterworth参数敏感性
- Figure 3: Drude参数敏感性 (New)
- Figure 4: 敏感性对比 (Updated)

---

## 7. 运行说明

在MATLAB中运行以下命令：

```matlab
cd /Users/mac/Desktop/thesis/thesis/research_output/20260118_parameter_sensitivity/code
sensitivity_analysis_lorentz_filter
```

代码将自动显示所有图表并输出数值结论。

---

## 8. 预期结论

### Lorentz模型
- $f_{res}$ 归一化敏感性 >> $\gamma$ 归一化敏感性
- **建议**：固定 $\gamma$ 为先验值（如 0.5 GHz），仅反演 $f_{res}$

### Drude模型
- $f_p$ 对群时延具有决定性影响（敏感性极高）
- $\nu_e$ 对群时延的影响为二阶效应（敏感性极低，比 $f_p$ 低1-2个数量级）
- **建议**：反演时必须固定 $\nu_e$，仅反演 $f_p$（即 $n_e$）

### Butterworth滤波器
- 根据条件数分析决定是否需要降维
- 若条件数过大，优先固定敏感性最低的参数（通常是 $N$）
