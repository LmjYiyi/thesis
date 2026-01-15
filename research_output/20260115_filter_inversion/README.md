# 滤波器参数反演研究文件夹

**研究日期**：2026-01-15  
**研究目标**：验证LM.m反演算法能否用于滤波器参数（F0, B, N）反演

---

## 📁 文件结构

```
20260115_filter_inversion/
├── README.md                          ← 本文件
├── report.md                          ← 完整技术报告
├── theory_notes.md                    ← 理论推导（初版）
├── code/                              ← 仿真代码
│   ├── verify_inversion_optimized.m   ← ✅ 推荐使用（最终优化版）
│   ├── _archive/                      ← 历史版本（归档）
│   │   ├── filter_inversion_LM.m      ← v1: 双参数反演
│   │   ├── filter_inversion_3param_LM.m ← v2: 三参数反演（纯理论）
│   │   ├── filter_inversion_LFMCW.m   ← v3: LFMCW方法（建模问题）
│   │   └── verify_inversion_algorithm.m ← v4: 简化验证
└── references/                        ← 理论分析文档
    └── N_sensitivity_analysis.md      ← 阶数N敏感性分析
```

---

## 🎯 研究结论

### ✅ 核心发现

**LM.m反演算法可以高精度反演滤波器参数**：

| 参数 | 反演精度 | 结果 |
|------|----------|------|
| **F0（中心频率）** | **0.26%** | 优秀 |
| **BW（带宽）** | **2.33%** | 优秀 |
| **N（阶数）** | **1.76%** | 优秀（可取整） |

### 🔑 关键技巧

参考 `thesis-code/LM.m` 的优化策略：

1. **能量加权**：`Weights = (W_raw/max(W_raw))^2`
2. **边缘增强**：增加边缘数据权重，解耦B和N
3. **参数归一化**：`scale=1e10`，防止梯度消失
4. **Levenberg-Marquardt算法**：更鲁棒的优化

---

## 🚀 快速开始

### 运行最终优化代码

```matlab
% 在MATLAB中运行
cd code
verify_inversion_optimized

% 预期输出：
% F0误差: < 0.3%
% BW误差: < 3%
% N误差: < 2%
```

### 代码说明

- **输入**：理论群时延 + 噪声（模拟测量）
- **处理**：加权LM反演算法
- **输出**：F0, BW, N 三个参数的估计值

---

## 📊 主要成果

### 1. 理论验证

✅ 证明了反演算法的有效性（使用理论群时延+噪声）

### 2. 参数敏感性分析

- F0通过峰值位置直接确定（最高精度）
- B和N存在耦合，通过边缘数据解耦
- 详见 [N_sensitivity_analysis.md](references/N_sensitivity_analysis.md)

### 3. LFMCW方法探索

⚠️ 完整LFMCW信号处理链反演失败，原因：
- 滤波器传递函数相位建模不准确
- 需要使用MATLAB内置滤波器或实测数据

---

## 📄 相关文档

| 文档 | 说明 |
|------|------|
| [report.md](report.md) | 完整技术报告 |
| [theory_notes.md](theory_notes.md) | 滤波器群时延理论 |
| [N_sensitivity_analysis.md](references/N_sensitivity_analysis.md) | 阶数N敏感性分析 |

---

## 🔬 后续工作

1. 使用VNA实测滤波器群时延数据
2. 将反演算法应用于真实滤波器诊断
3. 扩展到其他类型滤波器（Chebyshev, Elliptic等）

---

## 📞 技术支持

如有问题，请参考：
- 原算法：`thesis-code/LM.m`
- 滤波器参数：`resources/parameters.txt`
