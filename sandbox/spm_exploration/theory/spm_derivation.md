# 单粒子模型(SPM)数学推导与简化策略

## 1. 完整SPM方程系统

### 1.1 固相扩散方程 (Fick第二定律)
在球形颗粒中，锂离子浓度 $c_s(r,t)$ 满足：

$$
\frac{\partial c_s}{\partial t} = \frac{D_s}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial c_s}{\partial r} \right)
$$

**边界条件**：
1. 中心对称性：$\left. \frac{\partial c_s}{\partial r} \right|_{r=0} = 0$
2. 表面反应：$-D_s \left. \frac{\partial c_s}{\partial r} \right|_{r=R_p} = \frac{j}{a_s F}$

### 1.2 Butler-Volmer反应动力学
表面反应电流密度：

$$
j = i_0 \left[ \exp\left(\frac{\alpha_a F}{RT} \eta\right) - \exp\left(-\frac{\alpha_c F}{RT} \eta\right) \right]
$$

过电位：
$$
\eta = \phi_s - \phi_e - U(c_{s,surf}) - R_f j
$$

### 1.3 电化学参数
- $D_s$: 固相扩散系数 (m²/s)
- $R_p$: 颗粒半径 (m)
- $i_0$: 交换电流密度 (A/m²)
- $U(c_{s,surf})$: 开路电压 (V)

## 2. 简化策略 (48小时可行性验证)

### 2.1 核心简化：均匀浓度假设
**假设**：颗粒内部浓度均匀分布，仅考虑表面浓度变化

简化方程：
$$
\frac{dc_{s,avg}}{dt} = -\frac{3}{R_p} \frac{j}{a_s F}
$$

**物理意义**：忽略径向扩散梯度，只考虑平均浓度变化

### 2.2 Butler-Volmer线性化
**假设**：小电流条件下，Butler-Volmer可线性化

简化方程：
$$
j \approx \frac{i_0 F}{RT} \eta
$$

### 2.3 时间尺度分离
**观察**：扩散时间常数 $\tau_d = \frac{R_p^2}{D_s} \sim 10^3$ s
**策略**：使用准静态近似，每个时间步求解稳态扩散

## 3. 数值求解方案

### 3.1 离散化方法
采用有限差分法，将球形颗粒离散为N个壳层：

```python
# 离散网格
r = np.linspace(0, R_p, N)  # 径向位置
dr = R_p / (N-1)            # 网格间距

# 离散扩散算子
def diffusion_operator(c):
    # 使用中心差分
    d2c_dr2 = (c[2:] - 2*c[1:-1] + c[:-2]) / dr**2
    # 考虑球坐标的1/r²项
    r_mid = r[1:-1]
    return D_s * (d2c_dr2 + (2/r_mid) * (c[2:] - c[:-2])/(2*dr))
```

### 3.2 刚性ODE处理策略
**问题**：扩散方程刚性比 $\sim 10^6$
**方案**：使用隐式积分方法

1. **后退欧拉法**：无条件稳定
2. **Crank-Nicolson**：二阶精度，有条件稳定
3. **自适应步长**：根据梯度调整时间步

### 3.3 PyTorch实现要点
```python
# 关键实现细节
1. 使用torch.autograd计算物理约束梯度
2. 实现自定义PDE求解层
3. 梯度裁剪防止爆炸
4. 双精度浮点数提高稳定性
```

## 4. 可行性验证指标

### 4.1 成功标准
1. ✅ Loss稳定收敛（无NaN）
2. ✅ 单epoch耗时 < 5分钟（RTX 4060）
3. ✅ 内存占用 < 8GB
4. ✅ 梯度范数稳定

### 4.2 失败标志
1. ❌ 连续NaN出现
2. ❌ Loss震荡幅度 > 100%
3. ❌ 内存溢出
4. ❌ 训练进度停滞

## 5. 退路方案

如果SPM在24小时内无法稳定，切换到：
**宏观热力学ODE约束**：
$$
\frac{dC}{dt} = -k \cdot C \cdot \exp\left(-\frac{E_a}{RT}\right)
$$

**优势**：
1. 计算简单，O(1)复杂度
2. 物理意义明确（Arrhenius方程）
3. 已在实际工程中验证
4. 与Chronos-PINN兼容性好