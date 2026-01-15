# 时延计算方法修复报告

**日期**: 2026-01-14  
**修复范围**: `final_output/figure_code/` 中的3个绘图脚本

---

## ✅ 修复摘要

将简化的时延计算公式替换为**完整Drude模型（相位求导法）**，与 `thesis-code/LM.m` 保持一致。

---

## 📊 修复前后对比

### ❌ 修复前（简化公式）

```matlab
% 忽略碰撞频率的简化公式
tau_theory = (d/c) * (1 ./ sqrt(1 - (f_p./f_probe).^2) - 1);
```

**问题**：
- 仅考虑截止频率 `f_p`
- 忽略碰撞频率 `nu_e` 的影响
- 与反演算法 LM.m 不一致

---

### ✅ 修复后（完整Drude模型）

```matlab
% 使用完整Drude模型相位求导法
tau_theory = calculate_drude_delay(f_probe, n_e, nu_e, d, c, epsilon_0, m_e, e);

% 局部函数定义
function tau_rel = calculate_drude_delay(f_vec, ne_val, nu_val, d, c, eps0, me, e_charge)
    omega_vec = 2 * pi * f_vec;
    wp_val = sqrt(ne_val * e_charge^2 / (eps0 * me));
    
    % 复介电常数（含碰撞频率）
    eps_r = 1 - (wp_val^2) ./ (omega_vec .* (omega_vec + 1i*nu_val));
    
    % 复波数
    k_vec = (omega_vec ./ c) .* sqrt(eps_r);
    
    % 相位
    phi_plasma = -real(k_vec) * d;
    
    % 数值微分求群时延
    d_phi = diff(phi_plasma);
    d_omega = diff(omega_vec);
    tau_total = -d_phi ./ d_omega;
    tau_total = [tau_total, tau_total(end)];
    
    % 相对时延
    tau_rel = tau_total - (d/c);
end
```

**优势**：
- ✅ 完整考虑碰撞频率 `nu_e`
- ✅ 使用相位求导法（物理严格）
- ✅ 与 `thesis-code/LM.m` 的 `calculate_theoretical_delay` 函数完全一致
- ✅ 适用于精确计算和反演验证

---

## 📝 修复文件清单

| 文件 | 修复行数 | 修复内容 |
|------|---------|---------|
| `plot_fig_4_7.m` | 第48行 + 新增函数 | 三种电子密度工况的时延曲线计算 |
| `plot_fig_4_6.m` | 第78行 + 新增函数 | ESPRIT特征提取的理论曲线对比 |
| `plot_fig_4_10.m` | 第44、54行 + 新增函数 | 不同碰撞频率先验的拟合对比 |

---

## 🔬 理论依据

### Drude模型复介电常数

$$
\epsilon_r(\omega) = 1 - \frac{\omega_p^2}{\omega(\omega + i\nu_e)}
$$

### 群时延计算（相位求导法）

$$
\tau_g = -\frac{d\phi}{d\omega}
$$

其中相位为：

$$
\phi(\omega) = -\text{Re}[k(\omega)] \cdot d
$$

复波数为：

$$
k(\omega) = \frac{\omega}{c} \sqrt{\epsilon_r(\omega)}
$$

### 相对时延

$$
\tau_{\text{rel}} = \tau_g - \frac{d}{c}
$$

---

## ⚠️ 未修改的文件（已使用完整模型）

以下文件**已经使用完整Drude模型**，无需修改：

| 文件 | 计算方法 | 状态 |
|------|---------|------|
| `plot_fig_3_3a.m` | `calculate_drude_response` 函数 | ✅ 正确 |
| `plot_fig_3_3b.m` | `calculate_drude_response` 函数 | ✅ 正确 |
| `plot_fig_3_4.m` | 泰勒级数展开（专用于时变时延） | ✅ 正确 |
| `plot_fig_3_5.m` | 差频相位计算（二次项） | ✅ 正确 |
| `plot_fig_3_6.m` | 频谱展宽解析公式 | ✅ 正确 |
| `plot_fig_3_7.m` | 参数空间映射（非线性度因子） | ✅ 正确 |
| `plot_fig_4_3.m` | 修正项影响（含一阶导数） | ✅ 正确 |
| `plot_fig_4_4.m` | 权重分布（幅度计算） | ✅ 正确 |
| `plot_fig_4_5.m` | LM收敛轨迹（示意图） | ✅ 正确 |
| `plot_fig_4_8.m` | 残差分布（统计图） | ✅ 正确 |
| `plot_fig_4_9.m` | 鲁棒性测试（误差曲线） | ✅ 正确 |

---

## ✅ 验证检查清单

- [x] 修复了3个使用简化公式的文件
- [x] 添加了 `calculate_drude_delay` 函数（与 LM.m 一致）
- [x] 所有时延计算现在都考虑碰撞频率
- [x] 物理模型与反演算法保持一致
- [x] 代码注释清晰，标注了与 LM.m 的对应关系

---

## 🎯 结论

所有论文绘图脚本现在使用**统一的完整Drude模型**进行时延计算，确保与反演算法 `thesis-code/LM.m` 的物理建模完全一致。

碰撞频率 `nu_e = 1.5 GHz` 的影响已被正确纳入计算，消除了潜在的系统误差。
