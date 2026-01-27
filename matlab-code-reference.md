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
| `LM_MCMC.m` | **4.3 + 4.4** | Drude模型MCMC反演（理想无噪声版） |
| `LM_MCMC_with_noise.m` | **4.4** | Drude模型MCMC反演（含噪声鲁棒性测试）🆕 |
| `LM.m` | **4.4** 对比基线 | Drude模型LM点估计 |
| `CST_CSRR_Automation.m` | **5.1.1** | CSRR波导CST自动化建模 |
| `extract_lorentz_params_from_s21.m` | **5.1.2** | S21参数拟合Lorentz模型 |
| `LM_lorentz_CST_LFMCW.m` | **5.1.3** | CST数据LFMCW信号处理+MCMC |
| `LFMCW_filter_MCMC.m` | **5.2** | 滤波器三参数MCMC反演 |
| `initial.m` | **5.3.2** | 系统标定/4.4传统方法对比 |


## 终稿绘图代码索引 (final_output/figure_code)

> ⚠️ 注意：此目录下的代码为论文终稿专用绘图脚本，经过美化和格式统一。

| 代码文件 | 对应图号 | 功能描述 |
|:---|:---|:---|
| `plot_fig_3_2_0_matlab_theory.m` | **Fig 3.2** | 理论基准：群时延与频率关系 |
| `plot_fig_3_2_1_ne_sensitivity.m` | **Fig 3.2** | 参数分析：电子密度($n_e$)敏感性 |
| `plot_fig_3_2_2_nue_sensitivity.m` | **Fig 3.2** | 参数分析：碰撞频率($\nu_e$)敏感性 |
| `plot_fig_3_2_3_4_cst_density.m` | **Fig 3.2** | CST仿真：不同密度下的响应对比 |
| `plot_fig_3_2_5_multisolution.m` | **Fig 3.2** | 诊断困难：多解性问题可视化 |
| `plot_fig_3_2_time_frequency_spectrogram_SIMPLE.m` | **Fig 3.2** | 演示：时频图(Spectrogram)基本特征 |
| `plot_fig_3_4.m` | **Fig 3.4** | 色散效应：差频信号频谱畸变 |
| `plot_fig_3_4_spectrum_dispersion.m` | **Fig 3.4** | 补充：频谱色散细节 |
| `plot_fig_3_5_6_bandwidth_dispersion_effects.m` | **Fig 3.5-3.6** | 带宽效应：不同带宽下的去调频结果 |
| `plot_fig_3_7.m` | **Fig 3.7** | 工程判据：参数空间分布 |
| `plot_fig_3_7_bandwidth_zero_point.m` | **Fig 3.7** | 细节：零点带宽分析 |
| `plot_fig_3_7_criterion_parameter_space.m` | **Fig 3.7** | 细节：判据参数空间 |
| `plot_fig_4_1_sensitivity_comparison.m` | **Fig 4.1** | 诊断动机：高/低频波段敏感性对比 |
| `plot_fig_4_2_flat_valley_3d.m` | **Fig 4.2** | 反演难点：目标函数"平底谷"三维形貌 |
| `plot_fig_4_3_posterior_comparison.m` | **Fig 4.3** | 贝叶斯推断：先验与后验分布对比 |
| `plot_fig_4_4_spectrogram.m` | **Fig 4.4** | 预处理：含噪信号时频特征提取 |
| `plot_fig_4_5_mdl_criterion.m` | **Fig 4.5** | 模型选择：MDL准则阶数判定 |
| `plot_fig_4_6_feature_trajectory.m` | **Fig 4.6** | 过程可视化：MCMC马尔可夫链轨迹 |
| `plot_fig_4_7.m` | **Fig 4.7** | 验证I：Drude模型拟合效果(时域/频域) |
| `plot_fig_4_8.m` | **Fig 4.8** | 验证II：参数联合后验分布(Corner Plot) |
| `plot_fig_4_9.m` | **Fig 4.9** | 验证III：反演结果置信带与真值对比 |
| `plot_fig_4_10_4_11.m` | **Fig 4.10-4.11** | 复杂场景：非均匀/时变等离子体反演 |
| `plot_fig_4_12_robustness.m` | **Fig 4.12** | 性能评估：不同SNR下的算法鲁棒性 |
| `plot_fig_5_7_to_5_12_butterworth.m` | **Fig 5.7-5.12** | 普适性验证：Butterworth滤波器实验系列 |

---

## LM_MCMC_with_noise.m 详细参数（4.4节专用）

### 仿真参数表

| 参数符号 | 物理含义 | 数值设置 | 单位 |
|---------|---------|---------|------|
| $f_{\mathrm{start}}$ | 起始频率 | 34.2 | GHz |
| $f_{\mathrm{end}}$ | 终止频率 | 37.4 | GHz |
| $B$ | 信号带宽 | 3.2 | GHz |
| $T_m$ | 扫频周期 | 50 | μs |
| $f_c$ | 等离子体截止频率 | 33 | GHz |
| $n_e$ | 电子密度（由$f_c$计算） | 4.96×10¹⁸ | m⁻³ |
| $\nu_e$ | 碰撞频率 | 1.5 | GHz |
| $d$ | 等离子体厚度 | 150 | mm |
| SNR | 信噪比 | 20 | dB |

### 输出图表

| Figure编号 | 内容描述 | 论文图号建议 |
|-----------|---------|-------------|
| Figure 10 | 纯净信号 vs 含噪信号对比（时域波形） | 图4-X |
| Figure 11 | MCMC采样轨迹 (Trace Plot) | 图4-X |
| Figure 12 | 参数联合后验分布 (Corner Plot) | 图4-X |
| Figure 13 | MCMC拟合验证（测量点+后验均值曲线+95%置信带） | 图4-X |

### 关键输出变量

| 变量名 | 含义 | 示例值 |
|-------|------|-------|
| `ne_mean` | 电子密度后验均值 | ~4.83×10¹⁸ m⁻³ |
| `ne_std` | 电子密度后验标准差 | ~7.7×10¹⁷ m⁻³ |
| `nu_mean` | 碰撞频率后验均值 | ~2.3 GHz |
| `nu_std` | 碰撞频率后验标准差 | ~1.4 GHz |
| `accept_rate` | MCMC接受率 | ~51% |

### 噪声模型说明

详见 `thesis-code/噪声模型说明_论文写作指导.md`

---

## LM_MCMC.m vs LM_MCMC_with_noise.m 对比

| 特性 | LM_MCMC.m（理想版） | LM_MCMC_with_noise.m（含噪版） |
|-----|-------------------|------------------------------|
| 噪声模型 | 无 | 射频端AWGN (SNR=20dB) |
| 用途 | 验证算法正确性 | 验证算法鲁棒性 |
| 对应章节 | 4.3 方法论 | 4.4 仿真验证 |
| 结果可视化 | 理想曲线拟合 | 含置信带的不确定性量化 |
