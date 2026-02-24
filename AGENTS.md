# AGENTS.md

本仓库是一个**硕士毕业论文项目**（非传统软件工程项目），包含 MATLAB 仿真代码、Markdown 论文章节以及 AI 辅助写作/评审工作流系统。

**论文题目：**《基于LFMCW时延轨迹特征与参数反演的等离子体诊断及色散等效研究》

**研究领域：** LFMCW 雷达信号处理、等离子体诊断、贝叶斯反演（MCMC）

## 构建 / 运行 / 测试命令

本项目无传统构建系统，使用 MATLAB 和 Markdown 进行开发与写作。

### 运行 MATLAB 仿真

```bash
# 所有 MATLAB 脚本均可独立运行 —— 在 MATLAB 中打开后按 F5 执行
# 核心仿真脚本（位于 thesis-code/）：
#   LM_MCMC.m              — Drude 等离子体 MCMC 反演（核心）
#   LM_MCMC_with_noise.m   — 含噪声的 MCMC 鲁棒性测试
#   LM_lorentz_MCMC.m      — Lorentz 超材料 MCMC 反演
#   LFMCW_filter_MCMC.m    — 切比雪夫滤波器三参数 MCMC
#   LFMCW_filter_inversion_FINAL.m — 滤波器 LM 反演
#
# 绘图脚本（位于 final_output/figure_code/）：
#   命名格式：plot_fig_{章}_{节}_{描述}.m
#
# 环境要求：MATLAB R2018b 及以上，需安装信号处理工具箱和优化工具箱
```

### 运行单个仿真脚本

在 MATLAB 中打开 `.m` 文件后按 F5，或在命令窗口中执行：
```matlab
run('thesis-code/LM_MCMC.m')
```

### Markdown 转 Word

```bash
pandoc input.md -o output.docx
```

### 无测试套件、代码检查工具或 CI/CD 流水线

验证方式为目视检查 MATLAB 图形输出，并对比 `fprintf` 打印结果与已知参考值。

## 项目结构

```
.agent/workflows/     — AI 智能体工作流定义（6 个文件）
resources/            — 核心参考文档（角色、风格、公式模板、大纲）
thesis-code/          — MATLAB 仿真脚本（主要算法）
simulation/           — ADS/CST 仿真数据及处理脚本
output/               — 论文章节草稿（按版本号：_v1, _v2, _v3）
reviews/              — 专家评审报告（S/A/B/C 等级评分）
final_output/         — 定稿章节、图表及绘图代码
research_output/      — 按日期归档的科研探索成果
```

## 不可违背的理论约束

以下为已确立的物理结论，在任何论文文本或代码中**绝对不可违背**：

1. **碰撞频率是二阶微扰**：$\nu_e$ 对群时延的贡献量级为 $(\nu_e/\omega)^2$（二阶小量），而对幅度衰减的贡献是一阶主导。
2. **反演策略**：将 $\nu_e$ 固定为常数，仅反演 $n_e$。**绝不可**建议同时反演两者。
3. **非线性度因子**（分母必须是 1.5 次方）：
   $\eta(f) = \frac{B}{f} \cdot \frac{(f_p/f)^2}{[1-(f_p/f)^2]^{1.5}}$
4. **工程适用性判据**：$B \cdot \eta \cdot \tau_0 \le 1$
5. **技术路线**：滑动窗口 → MDL 信源估计 → ESPRIT 超分辨 → MCMC 贝叶斯反演

## MATLAB 代码规范

### 文件头部

每个脚本以装饰性注释块和初始化语句开头：
```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 脚本功能描述（中文）
% 用途：...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;
```

### 命名约定

| 类别 | 规则 | 示例 |
|------|------|------|
| 变量 | snake_case，结合物理符号 | `f_start`, `omega_p`, `n_e`, `tau_air` |
| 信号变量 | 时域用 `s_` 前缀，频域用 `S_` 前缀 | `s_tx`, `S_IF_air` |
| 函数 | snake_case | `calculate_theoretical_delay()`, `compute_log_likelihood()` |
| 优化器代价函数 | PascalCase（唯一例外） | `WeightedResiduals_Scaled()` |
| 特征数组 | `feature_` 前缀 | `feature_f_probe`, `feature_tau_absolute` |
| 算法脚本文件名 | PascalCase + 下划线 | `LM_MCMC.m`, `LM_lorentz.m` |
| 绘图脚本文件名 | `plot_fig_{章}_{节}_{描述}.m` | `plot_fig_4_1_sensitivity_comparison.m` |

### 文件内部结构

脚本按以下顺序组织：
1. 文件头注释块
2. `clc; clear; close all;`
3. 物理常数与参数设置
4. 信号生成
5. 传播仿真（频域处理）
6. 混频 / 差频信号提取
7. 高级信号处理（ESPRIT/MDL）—— `%% 7. ...` 节
8. 参数反演（LM 或 MCMC）—— `%% 9. ...` 节
9. 结果输出（`fprintf`）
10. 局部函数置于文件最末尾

### 错误处理

- **try/catch 配合哨兵值**用于优化回调函数中：
  - 失败时返回 `ones(size(f_data)) * 1e5` 作为大残差
  - 失败时返回 `-1e10` 作为极负对数似然
- **物理约束前置检查**（计算前提前返回）：
  ```matlab
  if ne_val <= 0, logL = -1e10; return; end
  ```
- **NaN/Inf 防护**在每次计算后执行：
  ```matlab
  if isnan(logL) || isinf(logL), logL = -1e10; end
  ```
- `error()` 仅用于不可恢复的致命数据问题。

### 注释规范

- 节标题：`%% N. 标题`（编号的单元分隔符）
- 小节分隔符：`% --- 描述 ---`
- 双语注释：中文用于物理解释，英文用于公式和 MATLAB 技术说明
- 进度日志：`fprintf('【步骤N】描述...\n');`

### 其他约定

- 数组预分配：循环前使用 `zeros(N_samples, 1)`
- 随机种子：随机采样前设置 `rng(42)` 确保可复现
- 频域安全处理：将 `omega == 0` 替换为 `1e-10` 避免除零
- 续行符：使用 `...`，后续行对齐缩进
- 物理常数在每个文件顶部重新定义（无共享常数文件）
- 长时间循环（ESPRIT/MCMC）使用 `waitbar` 显示进度
- 图形设置：`figure(N); clf; set(gcf, 'Position', [...]);`，字体使用 `'FontName', 'SimHei'`

## 论文写作规范

### 语言规则

- **禁止**使用第一人称（"我"、"我们"）
- **必须**使用"本文"、"本系统"、"该方法"、"实验表明"
- 因果状语前置："为了解决...，本文提出了..."
- 每章每节末尾必须有总结性段落

### 公式与术语

- 所有公式使用行内 LaTeX 格式：`$公式$`
- 公式格式参照 `resources/formula-templates.md`
- 核心术语（必须统一使用）：LFMCW、差频信号、群时延、栅栏效应、信噪比、MDL 准则、ESPRIT、Levenberg-Marquardt

### 版本管理

- 草稿：`output/第X章_X.X_标题_vN.md`（v1, v2, v3...）
- 评审：`reviews/第X章_X.X_标题_review_vN.md`
- 定稿：`final_output/第X章_X.X_标题_final.md`

### 交叉引用要求

撰写任何章节前**必须**检查：
1. `resources/核心研究档案.txt` —— 所有推导必须一致
2. `resources/大纲.txt` —— 章节结构已固定，不可更改
3. `final_output/` —— 与已定稿章节保持符号/术语一致性
4. 标有 `← [文件名].m` 的 MATLAB 代码 —— 参数必须与代码完全一致

## AI 工作流系统

`.agent/workflows/` 目录下定义了六个工作流：

| 工作流 | 文件 | 用途 |
|--------|------|------|
| 写作 | `thesis-writing.md` | 四种写作模式（初稿/修订/数据集成/自定义） |
| 评审 | `expert-review.md` | 七维度同行评审，S/A/B/C 等级评分 |
| 定稿 | `finalize.md` | 文件迁移、绘图代码生成、路径更新 |
| 科研 | `research-learning.md` | 五阶段科研探索流程 |
| 图表 | `figure-adjustment.md` | MATLAB 绘图代码修改（按风险等级分类） |
| 框架 | `framework-update.md` | 论文大纲更新与级联一致性检查 |

通过引用对应工作流文件来调用。AI 角色定义在 `resources/role.txt` 中 —— 设定为"985 高校博导"人设，对逻辑漏洞零容忍。

每次回答都需要在开头提示：Loxie你好
