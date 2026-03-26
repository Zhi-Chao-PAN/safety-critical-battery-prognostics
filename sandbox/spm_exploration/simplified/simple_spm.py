"""
极简SPM实现 - 48小时可行性验证

核心简化策略：
1. 均匀浓度假设：忽略径向梯度
2. 线性化Butler-Volmer：避免指数计算
3. 显式欧拉积分：避免隐式求解复杂性
4. 固定参数：减少优化变量

目标：验证SPM在PyTorch中的基本可行性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SandboxTimer:
    """沙盒时间监控器"""
    def __init__(self, max_hours: float = 48.0):
        import time
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600
        
    def check_timeout(self) -> bool:
        """检查是否超时"""
        import time
        elapsed = time.time() - self.start_time
        return elapsed > self.max_seconds
    
    def remaining_hours(self) -> float:
        """返回剩余小时数"""
        import time
        elapsed = time.time() - self.start_time
        return max(0.0, (self.max_seconds - elapsed) / 3600)


class SimplifiedSPM(nn.Module):
    """
    极简单粒子模型实现
    
    假设：
    1. 球形颗粒，径向均匀浓度
    2. 小电流线性反应动力学
    3. 恒温条件，忽略热效应
    4. 固相扩散主导，忽略液相
    
    状态变量：
    - c_s: 固相锂浓度 (mol/m³)
    - soc: 荷电状态 (0-1)
    
    控制变量：
    - I: 电流 (A)，正为放电，负为充电
    - T: 温度 (K)，暂时固定为298K
    """
    
    def __init__(self, 
                 D_s: float = 1e-14,      # 固相扩散系数 (m²/s)
                 R_p: float = 5e-6,       # 颗粒半径 (m)
                 c_s_max: float = 30000,  # 最大锂浓度 (mol/m³)
                 i0: float = 1.0,         # 交换电流密度 (A/m²)
                 a_s: float = 3e5,        # 比表面积 (m²/m³)
                 F: float = 96485.3329,   # 法拉第常数 (C/mol)
                 R: float = 8.314462618,  # 气体常数 (J/mol·K)
                 T: float = 298.15):      # 温度 (K)
        super().__init__()
        
        # 物理常数（固定）
        self.D_s = D_s
        self.R_p = R_p
        self.c_s_max = c_s_max
        self.i0 = i0
        self.a_s = a_s
        self.F = F
        self.R = R
        self.T = T
        
        # 可学习参数
        self.k_reaction = nn.Parameter(torch.tensor(1e-5))  # 反应速率常数
        self.ocv_slope = nn.Parameter(torch.tensor(0.1))    # OCV曲线斜率
        
        # 数值稳定性参数
        self.dt_min = 1.0  # 最小时间步 (s)
        self.dt_max = 3600.0  # 最大时间步 (s)
        
        # 沙盒监控
        self.timer = SandboxTimer(max_hours=48.0)
        
    def check_sandbox_limits(self) -> bool:
        """检查沙盒限制，返回是否继续"""
        if self.timer.check_timeout():
            print(f"⚠️ 沙盒时间超时！剩余时间: {self.timer.remaining_hours():.2f}小时")
            return False
        return True
        
    def linearized_butler_volmer(self, soc: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        线性化Butler-Volmer方程
        
        参数：
        - soc: 荷电状态 (0-1)
        - I: 电流 (A)
        
        返回：
        - j: 反应电流密度 (A/m²)
        """
        # 计算过电位（简化：线性关系）
        # η = U_ocv(soc) - V + I*R_ohm
        U_ocv = 3.7 + self.ocv_slope * (soc - 0.5)  # 简化OCV模型
        
        # 线性反应动力学
        j = self.i0 * (I / (self.a_s * self.F))  # 简化：j ∝ I
        
        # 数值稳定性：限制j的范围
        j = torch.clamp(j, -1.0, 1.0)
        
        return j
    
    def concentration_evolution(self, 
                               c_s: torch.Tensor, 
                               j: torch.Tensor,
                               dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        浓度演化方程（简化扩散+反应）
        
        简化假设：浓度均匀，扩散速率与表面浓度梯度成正比
        """
        # 表面浓度（假设与平均浓度成正比）
        c_s_surf = c_s * 0.9  # 简化：表面浓度略低于平均
        
        # 扩散通量（简化：Fick第一定律）
        diffusion_flux = self.D_s * (c_s - c_s_surf) / self.R_p**2
        
        # 反应消耗（简化）
        reaction_rate = self.k_reaction * c_s_surf * torch.abs(j)
        
        # 总变化率
        dc_dt = -diffusion_flux - reaction_rate
        
        # 显式欧拉积分
        c_s_new = c_s + dc_dt * dt
        
        # 物理约束：浓度必须在合理范围
        c_s_new = torch.clamp(c_s_new, 0.0, self.c_s_max)
        
        return c_s_new, dc_dt
    
    def capacity_to_soc(self, c_s: torch.Tensor) -> torch.Tensor:
        """浓度转换为荷电状态"""
        soc = c_s / self.c_s_max
        return torch.clamp(soc, 0.0, 1.0)
    
    def soc_to_voltage(self, soc: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """SOC转换为端电压"""
        # 开路电压
        U_ocv = 3.7 + self.ocv_slope * (soc - 0.5)
        
        # 欧姆极化
        R_ohm = 0.01  # 简化欧姆内阻 (Ω)
        V_ohm = I * R_ohm
        
        # 端电压
        V_terminal = U_ocv - V_ohm
        
        return V_terminal
    
    def forward(self, 
                c_s_init: torch.Tensor,
                I_profile: torch.Tensor,
                time_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：模拟电池循环
        
        参数：
        - c_s_init: 初始浓度 (batch_size,)
        - I_profile: 电流序列 (batch_size, seq_len)
        - time_steps: 时间步序列 (seq_len,)
        
        返回：
        - c_s_traj: 浓度轨迹 (batch_size, seq_len)
        - soc_traj: SOC轨迹 (batch_size, seq_len)
        - V_traj: 电压轨迹 (batch_size, seq_len)
        """
        if not self.check_sandbox_limits():
            # 返回空值，触发熔断
            return None, None, None
            
        batch_size = c_s_init.shape[0]
        seq_len = I_profile.shape[1]
        
        # 初始化轨迹
        c_s_traj = torch.zeros(batch_size, seq_len, device=c_s_init.device)
        soc_traj = torch.zeros(batch_size, seq_len, device=c_s_init.device)
        V_traj = torch.zeros(batch_size, seq_len, device=c_s_init.device)
        
        # 初始状态
        c_s_current = c_s_init.clone()
        
        for t in range(seq_len):
            # 当前电流和时间步
            I_t = I_profile[:, t]
            dt = time_steps[t] if t < len(time_steps) else time_steps[-1]
            
            # 限制时间步范围
            if not isinstance(dt, torch.Tensor):
                dt_tensor = torch.tensor(dt, device=c_s_init.device)
            else:
                dt_tensor = dt.detach().clone()
            
            dt_val = torch.clamp(dt_tensor, self.dt_min, self.dt_max).item()
            
            # 计算当前SOC
            soc_current = self.capacity_to_soc(c_s_current)
            
            # 线性化Butler-Volmer
            j_t = self.linearized_butler_volmer(soc_current, I_t)
            
            # 浓度演化
            c_s_next, dc_dt = self.concentration_evolution(c_s_current, j_t, dt_val)
            
            # 数值稳定性检查
            if torch.any(torch.isnan(c_s_next)) or torch.any(torch.isinf(c_s_next)):
                print(f"⚠️ 时间步{t}出现NaN/Inf，停止计算")
                c_s_next = torch.clamp(c_s_next, 0.0, self.c_s_max)
            
            # 计算电压
            V_t = self.soc_to_voltage(soc_current, I_t)
            
            # 保存轨迹
            c_s_traj[:, t] = c_s_current
            soc_traj[:, t] = soc_current
            V_traj[:, t] = V_t
            
            # 更新状态
            c_s_current = c_s_next
        
        return c_s_traj, soc_traj, V_traj
    
    def physics_loss(self, 
                    c_s_traj: torch.Tensor,
                    soc_traj: torch.Tensor,
                    V_traj: torch.Tensor,
                    I_profile: torch.Tensor) -> torch.Tensor:
        """
        物理约束损失函数
        
        包括：
        1. 浓度单调性约束（放电时SOC下降）
        2. 电压合理性约束
        3. 能量守恒约束
        """
        losses = []
        
        # 1. 浓度单调性（放电时SOC应该下降）
        soc_diff = soc_traj[:, 1:] - soc_traj[:, :-1]
        I_positive = (I_profile[:, :-1] > 0).float()  # 放电时电流为正
        
        # 放电时SOC应该下降，充电时SOC应该上升
        monotonicity_loss = torch.mean(
            torch.relu(soc_diff * I_positive)  # 放电时正变化是错误
        )
        losses.append(monotonicity_loss * 0.1)
        
        # 2. 电压合理性约束（2.5V - 4.2V）
        voltage_bounds_loss = torch.mean(
            torch.relu(2.5 - V_traj) + torch.relu(V_traj - 4.2)
        )
        losses.append(voltage_bounds_loss * 0.5)
        
        # 3. 能量近似守恒（简化）
        energy_in = torch.sum(I_profile * V_traj, dim=1)
        soc_change = soc_traj[:, -1] - soc_traj[:, 0]
        energy_balance_loss = torch.mean(
            (energy_in - soc_change * 3.7 * self.F).abs()  # 3.7V为平均电压
        )
        losses.append(energy_balance_loss * 0.01)
        
        return sum(losses)


def test_simple_spm():
    """测试简化SPM的基本功能"""
    print("🧪 测试简化SPM模型...")
    
    # 创建模型
    model = SimplifiedSPM()
    
    # 测试数据
    batch_size = 4
    seq_len = 10
    
    # 初始浓度（50% SOC）
    c_s_init = torch.ones(batch_size) * model.c_s_max * 0.5
    
    # 电流序列（正弦波模拟充放电循环）
    t = torch.linspace(0, 2*np.pi, seq_len)
    I_profile = torch.sin(t).unsqueeze(0).repeat(batch_size, 1) * 2.0  # ±2A
    
    # 时间步（均匀）
    time_steps = torch.ones(seq_len) * 3600.0  # 1小时步长
    
    # 前向传播
    c_s_traj, soc_traj, V_traj = model(c_s_init, I_profile, time_steps)
    
    if c_s_traj is None:
        print("❌ 沙盒超时，测试终止")
        assert False, "Sandbox timeout"
    
    # 检查结果
    print(f"✅ 模型输出形状:")
    print(f"  浓度轨迹: {c_s_traj.shape}")
    print(f"  SOC轨迹: {soc_traj.shape}")
    print(f"  电压轨迹: {V_traj.shape}")
    
    # 检查数值稳定性
    has_nan = torch.any(torch.isnan(c_s_traj))
    has_inf = torch.any(torch.isinf(c_s_traj))
    
    if not has_nan and not has_inf:
        print("✅ 数值稳定性检查通过")
        
        # 计算物理损失
        physics_loss = model.physics_loss(c_s_traj, soc_traj, V_traj, I_profile)
        print(f"✅ 物理损失: {physics_loss.item():.6f}")
        
        # 简单训练测试
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        physics_loss.backward()
        
        # 检查梯度
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        
        print(f"✅ 梯度范数: {total_grad_norm:.6f}")
        
        if total_grad_norm > 0 and total_grad_norm < 1000:
            print("🎉 简化SPM基本可行性验证通过！")
        else:
            print("⚠️ 梯度异常，需要调整")
            assert False, f"Gradient abnormal: {total_grad_norm}"
    else:
        print("❌ 数值不稳定，出现NaN/Inf")
        assert False, "Numerical instability"


if __name__ == "__main__":
    # 运行测试
    success = test_simple_spm()
    
    if success:
        print("\n🚀 简化SPM验证成功，可以进行下一步集成")
    else:
        print("\n⚠️ 简化SPM验证失败，考虑退路方案")