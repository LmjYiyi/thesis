# 滤波器参数反演研究文件夹

**研究日期**：2026-01-15 ~ 2026-01-16  
**研究目标**：验证LM.m反演算法能否用于滤波器参数（F0, B, N）反演  
**研究状态**：✅ **已完成并通过工程级验证**

---

## 📁 文件结构

```
research_output/20260115_filter_inversion/
├── README.md                      # 本文档(项目概览)
├── FINAL_REPORT.md               # 🎓 最终研究报告(完整版)
├── theory_notes.md                # 理论推导笔记
├── report.md                      # 初版可行性报告
│
├── code/                          # 核心代码
│   ├── LFMCW_filter_inversion_FINAL.m  # ⭐ 最终版(工程级,含非理想特性)
│   ├── verify_inversion_optimized.m    # ⭐ 优化验证版(理论数据)
│   └── _archive/                 # 历史版本归档
│       ├── README_ARCHIVE.md
│       ├── filter_inversion_LM.m          # v1: 双参数
│       ├── filter_inversion_3param_LM.m   # v2: 三参数
│       ├── filter_inversion_LFMCW.m       # v3: 失败版
│       └── verify_inversion_algorithm.m   # v4: 简化验证
│
└── references/                    # 参考资料
    ├── N_sensitivity_analysis.md      # 阶数N敏感性分析
    └── amplitude_parameters_impact.md # 🆕 幅度参数影响分析

```

---

## 📚 核心文档索引

| 文档 | 内容 | 阅读优先级 |
|------|------|-----------|
| **[FINAL_REPORT.md](FINAL_REPORT.md)** | 完整研究报告：理论、算法、实验、幅度参数分析、智能初始值等 | ⭐⭐⭐⭐⭐ |
| **[amplitude_parameters_impact.md](references/amplitude_parameters_impact.md)** | 专题：幅度参数如何通过权重机制影响反演 | ⭐⭐⭐⭐ |
| **[N_sensitivity_analysis.md](references/N_sensitivity_analysis.md)** | 阶数N对群时延曲线的影响分析 | ⭐⭐⭐ |
| **[theory_notes.md](theory_notes.md)** | 基础理论推导 | ⭐⭐ |

---

## 🎯 研究结论

### ✅ 核心发现

**LM.m反演算法可以高精度反演滤波器参数**：

#### 理论验证版（`verify_inversion_optimized.m`）

| 参数 | 反演精度 | 结果 |
|------|----------|------|
| **F0（中心频率）** | **0.26%** | 优秀 |
| **BW（带宽）** | **2.33%** | 优秀 |
| **N（阶数）** | **1.76%** | 优秀（可取整） |

#### 工程级LFMCW版（`LFMCW_filter_inversion_FINAL.m`）

| 参数 | 反演精度 | 结果 |
|------|----------|------|
| **F0（中心频率）** | **0.00%** | 完美 |
| **BW（带宽）** | **0.43%** | 极高 |
| **N（阶数）** | **0.50%** | 极高 |

**关键突破**：
1. ✅ 相位积分修正：$\phi(\omega) = -\int \tau_g d\omega$
2. ✅ 模型自洽：仿真源与反演模型完全匹配
3. ✅ 幅度参数建模：纹波、阻带抑制、底噪
4. ✅ 智能初始值：基于FWHM和边缘陡峭度

---

## 🔑 关键技巧

参考 `thesis-code/LM.m` 的优化策略：

| 优化项 | 实现方法 | 效果 |
|--------|----------|------|
| **能量加权** | `Weights = (W_raw/max(W_raw))^2` | 高SNR点权重大 |
| **边缘增强** | `edge_factor = 1 + 0.5*(1-tau_norm)^2` | 解耦B和N |
| **参数归一化** | `scale=1e10` | 防止梯度消失 |
| **智能初始值** | 基于峰值位置、FWHM、边缘陡峭度 | 收敛更快 |
| **幅度建模** | 纹波+阻带抑制+底噪 | 工程级真实性 |

---

## 🚀 快速开始

### 方案1：理论验证版（推荐用于算法验证）

```matlab
% 在MATLAB中运行
cd code
verify_inversion_optimized

% 预期输出：
% F0误差: < 0.3%
% BW误差: < 3%
% N误差: < 2%
```

### 方案2：工程级LFMCW版（推荐用于系统仿真）

```matlab
% 在MATLAB中运行
cd code
LFMCW_filter_inversion_FINAL

% 预期输出：
% F0误差: < 0.5%
% BW误差: < 1%
% N误差: < 1%
% 包含完整的LFMCW信号处理流程可视化（6个图像）
```

---

## 📊 研究亮点

### 1. 幅度参数的"隐藏"作用

**关键认识**：虽然幅度参数（纹波、阻带抑制、插入损耗）不改变**群时延真值**，但通过**权重机制**深刻影响算法的"视野"！

| 参数 | Datasheet | 对反演的影响 | 建模优先级 |
|------|-----------|-------------|-----------|
| **阻带抑制** | ≥80 dB | ⭐⭐⭐⭐⭐ 决定性 | **必须** |
| **通带纹波** | ≤1.2 dB | ⭐⭐⭐ 次要 | 建议 |
| **插入损耗** | ≤2.0 dB | ⭐ 极小 | 可选 |

详见：[幅度参数影响分析](references/amplitude_parameters_impact.md)

### 2. 智能初始值策略

**从LM.m学到的哲学**：从物理特征中提取先验信息

```matlab
F0_guess = X_fit(peak_idx);                    // 峰值位置
BW_guess = 2 * FWHM;                          // 半高宽
N_guess = round(3 + 2/edge_slope);            // 边缘陡峭度
```

### 3. LFMCW自洽建模

**两大修正**：
1. 相位 = 群时延的**积分**（不是乘积！）
2. 仿真源用理论公式构建，与反演模型**完全自洽**

---

## 🔬 后续工作

1. ✅ 使用VNA实测滤波器群时延数据
2. 将反演算法应用于真实滤波器诊断
3. 扩展到其他类型滤波器（Chebyshev, Elliptic等）
4. 双域反演（幅度+相位联合反演）

---

## 📞 技术支持

如有问题，请参考：
- 原算法：`thesis-code/LM.m`
- 滤波器参数：`resources/parameters.txt`
- 完整报告：[FINAL_REPORT.md](FINAL_REPORT.md)
