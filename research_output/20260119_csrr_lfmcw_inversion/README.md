# CSRR超材料LFMCW雷达诊断研究

**研究日期**：2026-01-19
**研究目标**：基于CST全波仿真验证LFMCW雷达诊断超材料Lorentz参数的可行性

## 文件清单
- `report.md` - 完整技术报告
- `cst_lorentz/` - 代码目录（符号链接到 thesis-code）
  - `CST_CSRR_Automation.m` - CST自动建模
  - `LM_lorentz_CST_LFMCW.m` - 信号处理与MCMC反演
  - `extract_lorentz_params_from_s21.m` - 参数提取验证
  - `data/data.s2p` - CST S参数数据

## 主要发现
1. ✅ 谐振频率反演误差 < 1%，验证了LFMCW方法对全波仿真数据的有效性
2. ✅ 建立了CST仿真→S参数→LFMCW处理→MCMC反演的完整闭环
3. ⚠️ ESPRIT有效点数较少（32个），MCMC接受率偏高（82%）

## 后续工作
- 优化ESPRIT窗口参数以增加有效点
- 调整MCMC步长使接受率降至20-50%
- 添加噪声敏感性分析
- 实验验证（VNA测量）
