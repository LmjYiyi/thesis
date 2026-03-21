# 右侧边缘提取与 `ex/` 目录整理记录

## 0. 2026-03-21 变更记录（S2P曲线叠加与坐标回调）

- `lib/plot_trajectory_result.m` 已叠加 `data/HXLBQ-DTA1329-1-1.s2p` 的 S21 群时延曲线（与 `plot_s2p_group_delay_curve.m` 同源计算逻辑）。
- 三个入口脚本横坐标范围统一为 `36.5-37.5 GHz`：
  - `extract_exp_trajectory_baseline.m`
  - `extract_exp_trajectory_data_driven.m`
  - `extract_exp_trajectory_mirror.m`
- `plot_s2p_group_delay_curve.m` 的显示窗口回调到 `36.5-37.5 GHz`。
- 绘图导出失败时改为 `warning`，不再中断轨迹图显示与诊断输出。

## 1. 当前数据驱动版本的主逻辑结论

- 当前主脚本是 `extract_exp_trajectory_data_driven.m`。
- 当前右侧重建核心模块是 `lib/refine_edge_segment.m`（原 `rebuild_right_edge_local.m`，已内联并重命名）。
- 当前右侧新增点来自原始数据窗口中的 `ESPRIT` 候选模态，不再直接使用左侧镜像先验。
- 当前 `refine_edge_segment.m` 中已经去掉了：
  - `build_left_prior(...)`
  - `f_left_query = 2 * f_center - f_query` 这类镜像坐标映射
  - 基于左侧镜像点的目标预测分支
- 因此，从"提取与重建主逻辑"来看，当前版本已经可以视为"纯右侧数据点驱动"。

## 2. 需要说明的边界

- 虽然提取主逻辑已经不再使用镜像先验，但绘图阶段仍会叠加一条外部参考曲线：
  - `lib/likely_filter_delay_curve.m`
- 这条曲线只用于图上参考，不参与右侧重建与点筛选。
- 如果后续文件名要强调"纯数据驱动版"，应明确：
  - 算法主流程是纯数据驱动
  - 图中红色参考曲线不是提取真值，只是工程参考

## 3. 为什么右侧边缘更难提取

当前理论判断是：

1. 右侧问题的主因不是简单的低信噪比。
2. 右侧问题的主因也不是高频采样混叠。
3. 主因是扫频末端边界效应叠加多配置不一致性增强。

更准确地说，右侧边缘是一个：

- 边界区
- 非平稳区
- 弱可分辨区

因此需要单独重建，而不能完全依赖与中段相同的直接提点策略。

## 4. 右侧困难成因诊断的当前结论

诊断脚本：

- `diagnose_edge_difficulty.m`（原 `diagnose_right_edge_difficulty.m`，已通用化并重命名）

在完成频率轴校准后，当前定量诊断支持以下结论。

### 4.1 幅度并不是主矛盾

- `left_edge` 的 `RMS = 0.080 mV`
- `flat_mid` 的 `RMS = 0.084 mV`
- `right_edge` 的 `RMS = 0.080 mV`

左右与中段的 `RMS` 差异不大，说明右侧并不是"信号幅度明显塌陷"，因此右侧提取困难不宜简单归因为低信噪比。

### 4.2 右侧最显著的问题是边界支撑不足

- `left_edge` 的 `boundary = 0.730`
- `flat_mid` 的 `boundary = 0.575`
- `right_shoulder` 的 `boundary = 0.300`
- `right_edge` 的 `boundary = 0.150`

`boundary` 越小，说明窗口越靠近扫频末端边界。右侧边缘的 `boundary` 最低，说明窗口有效支撑最差，边界截断效应在右侧最明显。

### 4.3 右侧最难的直接证据是多配置一致性恶化

多配置一致性统计如下：

- `left_edge`: `IQR = 0.227 ns`, `STD = 0.178 ns`
- `left_shoulder`: `IQR = 0.140 ns`, `STD = 0.143 ns`
- `flat_mid`: `IQR = 0.299 ns`, `STD = 0.223 ns`
- `right_shoulder`: `IQR = 0.360 ns`, `STD = 0.320 ns`
- `right_edge`: `IQR = 0.589 ns`, `STD = 0.885 ns`

其中 `right_edge` 的 `IQR` 与 `STD` 最大，说明同一点位在不同窗口长度、不同 `L_sub` 配置下的估计分散最严重。也就是说，右侧边缘不是"完全提不出来"，而是"不同配置下答案不稳定"，这是右侧必须单独重建的最强证据。

### 4.4 子空间可分辨性下降是辅助因素，不是唯一主因

- `left_edge` 的 `eigGap = 1.1`
- `flat_mid` 的 `eigGap = 1.7`
- `right_shoulder` 的 `eigGap = 1.2`
- `right_edge` 的 `eigGap = 1.1`

右侧的 `eigGap` 相比中段有所下降，说明子空间可分辨性变差。但其变化量没有一致性指标那么剧烈，因此更合理的结论是：

- 子空间可分辨性下降是助推因素
- 边界效应和多配置不一致性才是主导因素

### 4.5 当前诊断不支持"高频混叠主导"

从当前结果看：

- 幅度没有明显塌陷
- 右侧主要问题集中在边界支撑与配置一致性上
- 没有直接证据表明是奈奎斯特意义下的高频采样混叠

因此，当前更准确的表述应是：

- 右侧存在边界窗导致的谱泄漏
- 存在模态竞争与子空间失稳
- 但不应直接表述为"高频混叠是主因"

### 4.6 当前最稳妥的总括

> 右侧边缘提取困难的根本原因是扫频末端边界导致窗口有效支撑不足，使不同窗口配置下的群时延估计一致性显著恶化，同时子空间可分辨性有所下降；而信号幅度并未明显衰减，因此问题不宜归因于简单的低信噪比或采样混叠。

## 5. 目录结构（重构后）

重构为"入口脚本 + lib 函数库"架构。所有函数模块移入 `lib/` 子目录，入口脚本通过 `addpath('lib')` 引用。

```
ex/
  extract_exp_trajectory_baseline.m      # 入口：基线版（仅固定窗口）
  extract_exp_trajectory_mirror.m        # 入口：镜像版
  extract_exp_trajectory_data_driven.m   # 入口：纯数据驱动版
  diagnose_edge_difficulty.m             # 入口：边缘提取困难诊断
  plot_exp_spectrum.m                    # 入口：频谱绘图（独立，无 lib 依赖）
  note.md                               # 本文件
  data/                                  # 实测数据 CSV
  lib/                                   # 函数模块
    cfg_lowpass_filter.m                 #   数据集配置工厂
    run_trajectory_pipeline.m            #   主管线（加载→预处理→提取→校准→融合→重建）
    load_measured_dataset.m              #   数据加载
    preprocess_if_signal.m              #   预处理（去直流、叠加平均、降采样、滤波）
    esprit_extract.m                     #   ESPRIT/MDL 提取算法
    trajectory_postprocess.m             #   后处理（清洗、校准、融合、区域分类、导出）
    refine_edge_segment.m               #   边缘精细重建（数据驱动，原 rebuild_right_edge_local.m）
    rebuild_right_edge.m                 #   边缘镜像重建（镜像先验版）
    likely_filter_delay_curve.m          #   参考群时延曲线构造
    plot_trajectory_result.m             #   统一绘图与 TIFF 导出
    load_reference_group_delay.m         #   参考群时延加载（stub，预留 .s2p 接口）
```

### 已删除的旧文件

- `rebuild_right_edge_local.m` — 代码已内联到 `lib/refine_edge_segment.m`
- `diagnose_right_edge_difficulty.m` — 已被 `diagnose_edge_difficulty.m` 替代

### 数据目录中当前主线未直接使用、可考虑归档的候选

- `data/lowpassfilter_filter_1.csv`
- `data/lowpassfilter_filter_2.csv`
- `data/Trace_0005.csv`
- `data/Trace_0006.csv`

## 6. 命名规范

重构后命名已统一为"功能职责"风格：

| 文件 | 职责 |
|------|------|
| `cfg_lowpass_filter.m` | 数据集参数配置 |
| `run_trajectory_pipeline.m` | 管线编排 |
| `load_measured_dataset.m` | 数据加载 |
| `preprocess_if_signal.m` | 信号预处理 |
| `esprit_extract.m` | ESPRIT 提取算法 |
| `trajectory_postprocess.m` | 后处理（清洗/校准/融合/工具函数） |
| `refine_edge_segment.m` | 边缘数据驱动重建 |
| `rebuild_right_edge.m` | 边缘镜像重建 |
| `plot_trajectory_result.m` | 结果绘图与导出 |

入口脚本统一以 `extract_exp_trajectory_` 为前缀，通过配置覆盖选择不同模式。
