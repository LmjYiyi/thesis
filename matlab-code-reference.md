---
description: MATLAB仿真代码与论文章节的映射参考
---

# MATLAB 仿真代码参考指南

> 代码位置：`thesis-code/`

## 代码与论文章节映射

| 代码文件 | 对应论文章节 | 核心功能 |
|----------|-------------|----------|
| `test.m` | **3.2.3** 多解性问题 | 曲线相交现象展示 |
| `lianghua.m` | **3.4.2** 工程判据 | 工程判据 ξ = B·η·τ₀ |
| `nue.m` | **4.1.1** 参数敏感性 | 验证电子密度主导性 |
| `LM_MCMC.m` | **4.3 + 4.4** | Drude模型MCMC反演与验证 |
| `LM.m` | **4.4** 对比基线 | Drude模型LM点估计 |
| `CST_CSRR_Automation.m` | **5.1.1** | CSRR波导CST自动化建模 |
| `extract_lorentz_params_from_s21.m` | **5.1.2** | S21参数拟合Lorentz模型 |
| `LM_lorentz_CST_LFMCW.m` | **5.1.3** | CST数据LFMCW信号处理+MCMC |
| `LFMCW_filter_MCMC.m` | **5.2** | 滤波器三参数MCMC反演 |
| `initial.m` | **5.3.2** | 系统标定/传统方法对比 |
