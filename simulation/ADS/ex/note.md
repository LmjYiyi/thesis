# 右侧边缘提取与 `ex/` 目录整理记录

## 1. 当前数据驱动版本的主逻辑结论

- 当前主脚本是 `extract_exp_trajectory_data_driven.m`。
- 当前右侧重建核心模块是 `rebuild_right_edge_local.m`。
- 当前右侧新增点来自原始数据窗口中的 `ESPRIT` 候选模态，不再直接使用左侧镜像先验。
- 当前 `rebuild_right_edge_local.m` 中已经去掉了：
  - `build_left_prior(...)`
  - `f_left_query = 2 * f_center - f_query` 这类镜像坐标映射
  - 基于左侧镜像点的目标预测分支
- 因此，从“提取与重建主逻辑”来看，当前版本已经可以视为“纯右侧数据点驱动”。

## 2. 需要说明的边界

- 虽然提取主逻辑已经不再使用镜像先验，但绘图阶段仍会叠加一条外部参考曲线：
  - `likely_filter_delay_curve.m`
- 这条曲线只用于图上参考，不参与右侧重建与点筛选。
- 如果后续文件名要强调“纯数据驱动版”，应明确：
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

- `diagnose_right_edge_difficulty.m`

在完成频率轴校准后，当前定量诊断支持以下结论。

### 4.1 幅度并不是主矛盾

- `left_edge` 的 `RMS = 0.080 mV`
- `flat_mid` 的 `RMS = 0.084 mV`
- `right_edge` 的 `RMS = 0.080 mV`

左右与中段的 `RMS` 差异不大，说明右侧并不是“信号幅度明显塌陷”，因此右侧提取困难不宜简单归因为低信噪比。

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

其中 `right_edge` 的 `IQR` 与 `STD` 最大，说明同一点位在不同窗口长度、不同 `L_sub` 配置下的估计分散最严重。也就是说，右侧边缘不是“完全提不出来”，而是“不同配置下答案不稳定”，这是右侧必须单独重建的最强证据。

### 4.4 子空间可分辨性下降是辅助因素，不是唯一主因

- `left_edge` 的 `eigGap = 1.1`
- `flat_mid` 的 `eigGap = 1.7`
- `right_shoulder` 的 `eigGap = 1.2`
- `right_edge` 的 `eigGap = 1.1`

右侧的 `eigGap` 相比中段有所下降，说明子空间可分辨性变差。但其变化量没有一致性指标那么剧烈，因此更合理的结论是：

- 子空间可分辨性下降是助推因素
- 边界效应和多配置不一致性才是主导因素

### 4.5 当前诊断不支持“高频混叠主导”

从当前结果看：

- 幅度没有明显塌陷
- 右侧主要问题集中在边界支撑与配置一致性上
- 没有直接证据表明是奈奎斯特意义下的高频采样混叠

因此，当前更准确的表述应是：

- 右侧存在边界窗导致的谱泄漏
- 存在模态竞争与子空间失稳
- 但不应直接表述为“高频混叠是主因”

### 4.6 当前最稳妥的总括

> 右侧边缘提取困难的根本原因是扫频末端边界导致窗口有效支撑不足，使不同窗口配置下的群时延估计一致性显著恶化，同时子空间可分辨性有所下降；而信号幅度并未明显衰减，因此问题不宜归因于简单的低信噪比或采样混叠。

## 5. 对目录整理的判断

当前 `simulation/ADS/ex/` 目录中，主线需要保留并保证可运行的入口文件为：

- `extract_exp_trajectory_mirror.m`
- `extract_exp_trajectory_baseline.m`
- `extract_exp_trajectory_data_driven.m`
- `plot_exp_spectrum.m`

这些入口依赖的核心模块为：

- `esprit_extract.m`
- `trajectory_postprocess.m`
- `rebuild_right_edge.m`
- `rebuild_right_edge_local.m`
- `likely_filter_delay_curve.m`

当前主线无用、适合后续归档而非直接删除的候选包括：

- `compare_exp_data_quality.m`

数据目录中当前主线未直接使用、可考虑归档的候选包括：

- `data/lowpassfilter_filter_1.csv`
- `data/lowpassfilter_filter_2.csv`
- `data/Trace_0005.csv`
- `data/Trace_0006.csv`

## 6. 当前命名规范判断

- `esprit_extract.m`：负责提点
- `trajectory_postprocess.m`：负责洗点、校准、融合
- `rebuild_right_edge.m`：负责镜像版右侧重建
- `rebuild_right_edge_local.m`：负责纯右侧数据驱动的右侧重建

当前目录命名风格仍是混杂的：

- 有按算法命名的
- 有按处理动作命名的
- 有按输出目标命名的

如果后续要继续规范命名，建议统一成“流程阶段 + 处理对象”的方式。
