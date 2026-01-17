# 贝叶斯 MCMC 参数反演研究

**研究日期**：2026-01-18
**研究目标**：用 MCMC 替换 LM 算法，量化不敏感参数的不确定性

## 文件清单

| 文件 | 模型 | 状态 |
|-----|------|------|
| `code/LM_MCMC.m` | Drude (n_e, ν_e) | ✅ 完成 |
| `code/LM_lorentz_MCMC.m` | Lorentz (f_res, γ) | ✅ 完成 |
| `code/LFMCW_filter_MCMC.m` | Filter (F0, BW, N) | ✅ 完成 |

> **注意**：代码已同步至 `thesis-code/` 目录，可直接在该目录下运行。

## 核心改进

### 与 LM 算法对比

| 特性 | LM 算法 | MCMC 算法 |
|-----|---------|----------|
| 初始值 | 需人工猜测 | 从先验随机采样 |
| 输出 | 单点最优解 | 完整后验分布 |
| 不确定性 | 无法量化 | 自动量化 |
| 平底谷识别 | 无法识别 | 后验分布平坦即识别 |

## 使用说明

在 MATLAB 中运行：
```matlab
cd thesis-code
LM_MCMC          % 运行 Drude 模型 MCMC
LM_lorentz_MCMC  % 运行 Lorentz 模型 MCMC
LFMCW_filter_MCMC % 运行滤波器模型 MCMC
```

## 预期输出

- **Figure 11**: Trace plots (采样链收敛性)
- **Figure 12**: Corner plot (参数耦合可视化)
- **Figure 13**: 拟合验证 + 95% 置信带

## 核心结论

1. **Drude 模型**：$n_e$ 后验分布尖锐，$\nu_e$ 后验平坦 → 应固定 $\nu_e$
2. **Lorentz 模型**：$f_{res}$ 可精准反演，$\gamma$ 存在平底谷 → 应固定 $\gamma$
3. **滤波器模型**：$F_0$ 高精度，$N$ 分布宽广 → 建议固定 $N$

## 相关更新

- ✅ 核心研究档案新增"第六部分：贝叶斯不确定性量化"
- ✅ matlab-code-reference.md 已更新代码映射表
- ✅ 代码已复制至 thesis-code/ 目录
