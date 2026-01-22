---
description: MATLAB仿真代码与论文章节的映射参考
---

# MATLAB 仿真代码参考指南

> 代码位置：`thesis-code/`

---

## 代码目录

```
thesis-code/
├── LM.m                        # Drude模型LM点估计 (第四章对比基线)
├── LM_MCMC.m                   # Drude模型MCMC反演 (第四章核心)
├── LM_lorentz.m                # Lorentz超材料反演 - LM算法
├── LM_lorentz_MCMC.m           # Lorentz超材料反演 - MCMC算法
├── LFMCW_filter_inversion_FINAL.m  # 滤波器反演 - LM算法
├── LFMCW_filter_MCMC.m         # 滤波器反演 - MCMC算法 (第五章)
├── initial.m                   # 传统诊断方法/系统标定 (5.3节)
├── nue.m                       # 参数敏感性分析 (4.1.1节)
├── lianghua.m                  # 工程判据可视化 (3.4.2节)
├── test.m                      # 多解性问题/曲线相交 (3.2.3节)
├── lorentz_sensitivity_analysis.m  # Lorentz敏感性分析
├── cst_lorentz/                # CST仿真代码目录 (第五章)
│   ├── CST_SRR_Automation.m        # CST自动化建模 (5.1.1节)
│   ├── extract_lorentz_params_from_s21.m  # S21参数拟合 (5.1.2节)
│   └── LM_lorentz_CST_LFMCW.m      # CST+LFMCW反演 (5.1.3节)
└── README.md                   # 代码文档说明
```

---

## 代码与论文章节映射

| 代码文件 | 对应论文章节 | 核心功能 | 生成图表 |
|----------|-------------|----------|---------| 
| `test.m` | **3.2.3** 多解性问题 | 曲线相交现象展示 | 图 3-X |
| `lianghua.m` | **3.4.2** 工程判据 | 工程判据 $\xi = B \cdot \eta \cdot \tau_0$ | 工程边界图 |
| `nue.m` | **4.1.1** 参数敏感性 | 验证电子密度主导性 + 碰撞频率解耦 | 图 4-1a, 4-1b |
| `LM_MCMC.m` | **4.3 + 4.4** | Drude模型MCMC反演与验证 | 图 4-X (Trace, Corner) |
| `LM.m` | **4.4** 对比基线 | Drude模型LM点估计 | 图 4-X |
| `CST_SRR_Automation.m` | **5.1.1** | SRR波导CST自动化建模 | - |
| `extract_lorentz_params_from_s21.m` | **5.1.2** | S21参数拟合Lorentz模型 | 图 5-X |
| `LM_lorentz_CST_LFMCW.m` | **5.1.3** | CST数据LFMCW信号处理+MCMC | 图 5-X |
| `LFMCW_filter_MCMC.m` | **5.2** | 滤波器三参数MCMC反演 | 3×3 Corner Plot |
| `initial.m` | **5.3.2** | 系统标定/传统方法对比 | 图 5-X |

---

## 快速运行命令

```matlab
% === 第三章图表 ===
>> test         % 生成图 3-X (多解性问题/曲线相交)
>> lianghua     % 生成工程判据图 (3.4.2节)

% === 第四章图表 ===
>> nue          % 生成图 4-1a, 4-1b (参数敏感性)
>> LM_MCMC      % 生成 Drude MCMC 反演结果 (Trace, Corner, Fit)

% === 第五章图表 ===
>> cd cst_lorentz
>> LM_lorentz_CST_LFMCW  % CST Lorentz验证
>> cd ..
>> LFMCW_filter_MCMC     % 滤波器三参数MCMC验证

% === 对比分析 ===
>> initial      % 生成传统方法结果 (作为对比基线)
```

---

## 代码与公式对应

| 论文公式 | 代码位置 | 实现方式 |
|---------|---------|---------|
| (3-1) 复介电常数 | `LM_MCMC.m` | `eps_r = 1 - wp^2/(omega*(omega+1i*nu))` |
| (3-7) 群时延解析式 | `nue.m` | `tau_g = -diff(phi)./diff(omega)` |
| (3-18) 非线性度因子 η | `lianghua.m` | `eta = (B./f) .* (x.^2) ./ ((1-x.^2).^1.5)` |
| 工程判据 ξ | `lianghua.m` | `Xi = B .* eta .* tau0` |
| 曲线相交条件 | `test.m` | 多参数组合时延曲线对比 |
| MDL 准则 | `LM_MCMC.m` 7.3节 | `mdl_cost(k+1) = term1 + term2` |
| 加权似然函数 | `LM_MCMC.m` 9节 | `logL = -0.5 * sum(w .* residuals.^2)` |
| MCMC接受概率 | `LM_MCMC.m` 9.4节 | `log_alpha = logL_proposed - logL_current` |

---

## 论文引用模板

### 描述仿真设置
> 为了验证上述理论推导，本文基于 MATLAB 搭建了数值仿真平台。仿真参数设置如下：等离子体厚度 $d = 150$ mm，截止频率 $f_p = 29$ GHz（对应电子密度 $n_e = 1.04 \times 10^{19}$ m$^{-3}$），碰撞频率 $\nu_e = 1.5$ GHz。

### 描述多解性问题
> 图 3-X 展示了不同参数组合下群时延曲线的相交现象。由图可知，当电子密度与碰撞频率取不同组合时，时延曲线可能在某些频点相交，表明单点测量无法唯一确定介质参数。这一多解性问题论证了曲线拟合反演的必要性。

### 描述MCMC反演
> 为量化参数的不确定性，采用 Metropolis-Hastings 采样进行贝叶斯反演。图 4-X 给出了后验分布的 Corner Plot，可以观察到电子密度 $n_e$ 的后验分布呈尖锐单峰，而碰撞频率 $\nu_e$ 的后验分布相对平坦，验证了 4.1 节关于参数敏感性差异的分析。

---

## 研究成果目录索引

| 研究成果 | 目录 | 对应论文章节 |
|----------|------|-------------|
| 滤波器反演 | `research_output/20260115_filter_inversion/` | 5.2 |
| Lorentz敏感性 | `research_output/20260115_lorentz_sensitivity/` | 5.1.4 |
| 贝叶斯反演 | `research_output/20260118_bayesian_inversion/` | 4.3 |
| CST时域仿真 | `research_output/20260119_cst_timedomain_lfmcw/` | 5.1 |
| SRR LFMCW反演 | `research_output/20260119_csrr_lfmcw_inversion/` | 5.1 |
