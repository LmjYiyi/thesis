# 研究代码索引

本文档整理`thesis-code/`目录与`research_output/`目录的对应关系。

---

## 📂 目录结构对应关系

### 1. 滤波器参数反演研究

#### thesis-code → research_output

| thesis-code | research_output | 说明 |
|-------------|-----------------|------|
| `filter_inversion_feasibility.m` | `20260115_filter_inversion/code/_archive/` | **v0**: 可行性验证（双参数） |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/filter_inversion_LM.m` | **v1**: 双参数优化版 |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/_archive/filter_inversion_3param_LM.m` | **v2**: 三参数初版 |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/_archive/filter_inversion_LFMCW.m` | **v3**: LFMCW失败版 |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/_archive/verify_inversion_algorithm.m` | **v4**: 简化验证 |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/verify_inversion_optimized.m` | ⭐ **最终理论版** |
| *(直接在研究中开发)* | `20260115_filter_inversion/code/LFMCW_filter_inversion_FINAL.m` | ⭐ **最终工程版** |

**研究报告**：[`research_output/20260115_filter_inversion/FINAL_REPORT.md`](../research_output/20260115_filter_inversion/FINAL_REPORT.md)

---

### 2. Lorentz敏感性分析

#### thesis-code → research_output

| thesis-code | research_output | 说明 |
|-------------|-----------------|------|
| `lorentz_sensitivity_analysis.m` | `20260115_lorentz_sensitivity/` | Lorentz模型敏感性分析 |
| `LM_lorentz.m` | *(辅助代码，未归档)* | Lorentz反演中间版本 |

**研究报告**：[`research_output/20260115_lorentz_sensitivity/report.md`](../research_output/20260115_lorentz_sensitivity/report.md)

---

## 📝 研究演进时间线

### 滤波器参数反演

```
2026-01-15 上午
├─ filter_inversion_feasibility.m (v0)
│  └─ 证明可行性：F0<1%, B<5%
│
2026-01-15 中午
├─ filter_inversion_LM.m (v1)
│  └─ 加入LM.m优化技巧
│
2026-01-15 下午
├─ filter_inversion_3param_LM.m (v2)
│  └─ 新增N参数反演（初版，误差>10%)
│
├─ filter_inversion_LFMCW.m (v3)
│  └─ LFMCW尝试失败（相位建模错误）
│
├─ verify_inversion_algorithm.m (v4)
│  └─ 简化验证，确认算法本身正确
│
2026-01-15 晚上
├─ verify_inversion_optimized.m (最终理论版)
│  └─ 优化后：F0<0.3%, B<2.5%, N<2%
│
2026-01-16 凌晨
└─ LFMCW_filter_inversion_FINAL.m (最终工程版)
   ├─ 修正相位积分错误
   ├─ 模型自洽
   ├─ 加入幅度参数建模
   └─ 精度：F0=0%, B<0.5%, N<1%
```

---

## 🎯 代码使用建议

### 如果您想...

| 目标 | 推荐代码 | 位置 |
|------|---------|------|
| **理解研究起点** | `filter_inversion_feasibility.m` | `research_output/.../code/_archive/` |
| **验证算法精度** | `verify_inversion_optimized.m` | `research_output/.../code/` |
| **仿真完整系统** | `LFMCW_filter_inversion_FINAL.m` | `research_output/.../code/` |
| **查看失败案例** | `filter_inversion_LFMCW.m` | `research_output/.../code/_archive/` |
| **理解演进过程** | 阅读归档目录的README | `research_output/.../code/_archive/README_ARCHIVE.md` |

---

## ⚠️ 未归档的thesis-code文件

| 文件 | 状态 | 建议 |
|------|------|------|
| `LM_lorentz.m` | ❌ 未归档 | 可能是Lorentz研究的辅助代码，建议检查是否需要归档到`20260115_lorentz_sensitivity/` |

---

**最后更新**：2026-01-16  
**维护者**：Antigravity
