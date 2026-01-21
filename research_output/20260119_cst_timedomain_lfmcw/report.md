# CST时域仿真与LFMCW信号集成研究

**研究日期**：2026-01-19
**研究目标**：通过MATLAB控制CST进行时域仿真，实现LFMCW激励信号注入、时域信号导出和参数反演

---

## 1. 研究背景

### 1.1 当前方法 vs 目标方法

| 特性 | 当前方法（频域S参数） | 目标方法（时域仿真） |
|------|----------------------|---------------------|
| **激励** | 宽带高斯脉冲 | LFMCW调频信号 |
| **传递函数** | S21(f) 频域乘法 | 时域卷积/CST直接仿真 |
| **信号处理** | MATLAB后处理 | CST内部+MATLAB混合 |
| **物理真实性** | 中等（线性假设） | 高（含非线性效应） |
| **计算成本** | 低 | 高 |

### 1.2 研究动机

1. **验证线性假设**：频域方法假设系统线性，时域仿真可验证此假设
2. **捕获瞬态效应**：LFMCW信号的瞬态响应可能包含额外信息
3. **更真实的噪声模型**：时域仿真可添加真实噪声
4. **端到端验证**：建立完整的CST→MATLAB闭环

---

## 2. 理论分析

### 2.1 CST时域求解器

CST Microwave Studio提供两种主要求解器：
- **频域求解器（Frequency Domain Solver）**：基于有限元法，计算S参数
- **时域求解器（Time Domain Solver）**：基于FDTD/FIT，计算瞬态场

时域求解器的输入输出：
```
输入：端口激励信号 s_tx(t)
输出：端口输出信号 s_rx(t)
```

### 2.2 LFMCW激励信号

标准LFMCW信号：
$$s_{TX}(t) = \cos\left(2\pi f_0 t + \pi K t^2\right)$$

参数设计：
- 起始频率 $f_0 = 34.2$ GHz
- 带宽 $B = 3.2$ GHz  
- 扫频周期 $T_m = 50$ μs
- 调频斜率 $K = B/T_m = 64$ MHz/μs

### 2.3 信号流程

```mermaid
flowchart LR
    A[MATLAB生成LFMCW] --> B[写入CST激励文件]
    B --> C[CST时域仿真]
    C --> D[导出时域信号]
    D --> E[MATLAB混频处理]
    E --> F[ESPRIT特征提取]
    F --> G[MCMC参数反演]
```

---

## 3. 技术实现

### 3.1 CST激励信号定义

CST支持用户自定义激励信号，通过以下方式：

**方法1：User Defined Excitation Signal（VBA）**
```vba
With Solver
    .ExcitationType "UserDefined"
    .ExcitationSignalFile "lfmcw_excitation.sig"
End With
```

**方法2：ASCII信号文件格式**
```
% time (ns)  amplitude
0.000        0.000000
0.001        0.587785
0.002        0.951057
...
```

### 3.2 时域求解器配置

```vba
With Solver
    .Method "Hexahedral"
    .CalculationType "TD-S"  ' 时域S参数
    .StimulationPort "1"
    .SteadyStateLimit "-40"  ' dB
    .UseArfilter "True"
    .ArMaxEnergyDeviation "0.1"
End With
```

### 3.3 信号导出

```vba
With ASCIIExport
    .FileName "output_port2.txt"
    .Execute
End With
```

---

## 4. 代码实现计划

### 4.1 文件结构

```
thesis-code/cst_lorentz/
├── CST_CSRR_Automation.m        # 现有：频域建模
├── CST_TimeDomain_LFMCW.m       # 新增：时域仿真控制
├── generate_lfmcw_excitation.m  # 新增：生成激励文件
├── process_cst_timedomain.m     # 新增：处理时域输出
└── data/
    ├── data.s2p                 # 现有：频域数据
    ├── lfmcw_excitation.sig     # 新增：激励信号
    └── timedomain_output.txt    # 新增：时域输出
```

### 4.2 核心代码模块

#### 模块1：激励信号生成

```matlab
function generate_lfmcw_excitation(filename, f_start, f_end, T_m, f_s)
    % 生成LFMCW激励信号并保存为CST格式
    
    K = (f_end - f_start) / T_m;
    t = 0 : 1/f_s : T_m;
    
    % LFMCW信号
    phi = 2*pi*f_start*t + pi*K*t.^2;
    s = cos(phi);
    
    % 保存为CST ASCII格式 (时间单位：ns)
    fid = fopen(filename, 'w');
    for i = 1:length(t)
        fprintf(fid, '%.6e  %.6e\n', t(i)*1e9, s(i));
    end
    fclose(fid);
end
```

#### 模块2：CST时域仿真控制

```matlab
function CST_run_timedomain(mws, excitation_file, output_file)
    % 配置时域求解器
    vba_solver = [...
        'With Solver' newline ...
        '  .Method "Hexahedral"' newline ...
        '  .CalculationType "TD-S"' newline ...
        '  .StimulationPort "1"' newline ...
        '  .ExcitationType "UserDefined"' newline ...
        sprintf('  .ExcitationSignalFile "%s"', excitation_file) newline ...
        'End With'];
    invoke(mws, 'AddToHistory', 'configure TD solver', vba_solver);
    
    % 运行仿真
    solver = invoke(mws, 'Solver');
    invoke(solver, 'Start');
    
    % 导出信号
    % ...
end
```

#### 模块3：混频处理

```matlab
function [s_if, t_if] = process_timedomain(s_tx, s_rx, f_s, fc_lp)
    % 混频
    s_mix = s_tx .* s_rx;
    
    % 低通滤波
    [b, a] = butter(4, fc_lp/(f_s/2));
    s_if = filtfilt(b, a, s_mix);
end
```

---

## 5. 工程挑战

### 5.1 计算时间

| 仿真类型 | 预计时间 | 说明 |
|----------|---------|------|
| 频域（当前） | ~5分钟 | 单次扫频 |
| 时域（目标） | ~数小时 | 长时间瞬态仿真 |

**优化策略**：
- 降低仿真精度（网格稀疏化）
- 缩短扫频周期（$T_m$ 从50μs减至5μs）
- 使用AR滤波提前终止

### 5.2 数据量

LFMCW信号采样率80 GHz，50μs周期：
- 采样点数：$N = 80\times10^9 \times 50\times10^{-6} = 4\times10^6$
- 数据量：~32 MB（双精度）

**解决方案**：
- 降采样后导出
- 仅保存端口信号（不保存场分布）

### 5.3 CST COM接口限制

CST COM接口在某些版本中可能：
- 不支持完整的时域求解器控制
- 信号导出功能受限

**后备方案**：
- 使用CST内置脚本（VBA宏）
- 手动触发仿真，自动读取结果

---

## 6. 实施计划

### 第一阶段：可行性验证（2小时）

- [ ] 在CST GUI中手动配置时域求解器
- [ ] 测试用户自定义激励信号功能
- [ ] 验证信号导出格式

### 第二阶段：MATLAB集成（4小时）

- [ ] 编写`generate_lfmcw_excitation.m`
- [ ] 扩展`CST_CSRR_Automation.m`添加时域支持
- [ ] 编写信号后处理模块

### 第三阶段：闭环验证（4小时）

- [ ] 端到端测试
- [ ] 与频域方法结果对比
- [ ] 参数反演精度评估

---

## 7. 预期成果

1. **代码**：
   - `CST_TimeDomain_LFMCW.m` - 完整的时域仿真控制脚本
   - `generate_lfmcw_excitation.m` - 激励信号生成器

2. **数据**：
   - CST时域仿真输出信号
   - 与频域方法的对比结果

3. **验证**：
   - 时域方法反演精度 vs 频域方法
   - 计算效率对比

---

## 8. 结论

### 8.1 可行性评估

| 方面 | 评估 | 备注 |
|------|------|------|
| CST支持 | ✅ 可行 | 需要时域求解器许可 |
| MATLAB集成 | ⚠️ 部分 | COM接口可能受限 |
| 计算成本 | ⚠️ 较高 | 需优化参数 |
| 物理意义 | ✅ 有价值 | 验证线性假设 |

### 8.2 建议

> [!IMPORTANT]
> **推荐采用混合方案**：
> - **建模**：MATLAB控制CST创建模型
> - **激励**：通过文件传递LFMCW信号
> - **仿真**：CST GUI手动触发（避免COM限制）
> - **后处理**：MATLAB自动读取并处理

### 8.3 后续工作

1. 首先手动在CST中验证时域仿真可行性
2. 确认信号格式和导出方法
3. 再进行MATLAB自动化集成

---

## 附录

### A. CST时域求解器VBA参考

```vba
' 完整的时域求解器配置示例
Sub ConfigureTimeDomainSolver()
    With Solver
        .Method "Hexahedral"
        .CalculationType "TD-S"
        .StimulationPort "1"
        .SteadyStateLimit "-40"
        .UseArfilter "True"
        .ArMaxEnergyDeviation "0.1"
        .TimeBetweenUpdates "10"
        .AutoNormImpedance "True"
        .NormingImpedance "50"
    End With
    
    ' 用户自定义激励
    With Solver
        .ExcitationType "UserDefined"
        .ExcitationSignalFile "lfmcw.sig"
    End With
End Sub
```

### B. 参考资料

- [CST_CSRR_Automation.m](file:///c:/Users/admin/Desktop/lunwen/master_study/thesis/thesis-code/cst_lorentz/CST_CSRR_Automation.m) - 现有频域自动化脚本
- [LM_lorentz_CST_LFMCW.m](file:///c:/Users/admin/Desktop/lunwen/master_study/thesis/thesis-code/cst_lorentz/LM_lorentz_CST_LFMCW.m) - LFMCW信号处理参考
