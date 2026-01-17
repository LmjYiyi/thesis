# 参数敏感性分析研究

**研究日期**：2026-01-18
**研究目标**：分析 Lorentz 模型与 Butterworth 滤波器模型的参数敏感性，确定降维策略

## 文件清单

- `code/sensitivity_analysis_lorentz_filter.m` - 完整仿真代码

## 使用说明

在MATLAB中运行：
```matlab
cd code
sensitivity_analysis_lorentz_filter
```

## 预期结论

### Lorentz模型
- $f_{res}$ 敏感性远大于 $\gamma$
- 建议策略：固定 $\gamma$ 为先验值，仅反演 $f_{res}$

### Drude模型
- $f_p$ 对群时延影响显著，$\nu_e$ 影响微弱（二阶效应）
- 建议策略：固定 $\nu_e$，仅反演 $f_p$

### Butterworth滤波器
- 三参数 $(F_0, BW, N)$ 的敏感性量化
- 基于Jacobian条件数决定是否降维
