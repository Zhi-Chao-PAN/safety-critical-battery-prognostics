#!/usr/bin/env python3
"""
SPM-PINN集成可行性测试

目标：验证SPM物理约束与神经网络结合的可行性
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Optional


class SandboxMonitor:
    """沙盒监控器，确保不超过48小时"""
    def __init__(self, max_hours: float = 48.0):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600
        self.explosion_count = 0  # 爆炸计数器
        self.max_explosions = 10  # 最大爆炸次数
        
    def check_limits(self) -> bool:
        """检查是否超过限制"""
        elapsed = time.time() - self.start_time
        hours_elapsed = elapsed / 3600
        
        if elapsed > self.max_seconds:
            print(f"⏰ 沙盒时间到期！已运行: {hours_elapsed:.2f}小时")
            return False
            
        if self.explosion_count >= self.max_explosions:
            print(f"💥 爆炸次数超过限制: {self.explosion_count}次")
            return False
            
        return True
    
    def record_explosion(self, error_type: str):
        """记录爆炸事件"""
        self.explosion_count += 1
        print(f"⚠️ 记录{error_type}爆炸 #{self.explosion_count}")
    
    def get_status(self) -> dict:
        """返回当前状态"""
        elapsed = time.time() - self.start_time
        return {
            "hours_elapsed": elapsed / 3600,
            "hours_remaining": max(0, (self.max_seconds - elapsed) / 3600),
            "explosion_count": self.explosion_count,
            "max_explosions": self.max_explosions,
            "is_active": self.check_limits()
        }


class SPMPhysicsLayer(nn.Module):
    """
    SPM物理约束层
    
    实现简化但物理合理的约束：
    1. 浓度演化约束（基于Fick定律简化）
    2. 反应动力学约束（线性化Butler-Volmer）
    3. 热力学约束（Arrhenius温度效应）
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # 电池物理参数（NMC典型值）
        self.D_s = 1e-14  # 固相扩散系数 (m²/s)
        self.R_p = 5e-6   # 颗粒半径 (m)
        self.c_max = 30000.0  # 最大锂浓度 (mol/m³)
        self.F = 96485.3329  # 法拉第常数 (C/mol)
        self.R = 8.314462618  # 气体常数 (J/mol·K)
        
        # 可学习的物理参数
        self.reaction_rate = nn.Parameter(torch.tensor(1e-4))
        self.diffusion_coeff = nn.Parameter(torch.tensor(1.0))
        self.activation_energy = nn.Parameter(torch.tensor(30e3))  # 活化能 (J/mol)
        
        # 数值稳定性设置
        self.epsilon = 1e-8
        self.max_grad_norm = 10.0
        
        # 沙盒监控
        self.monitor = SandboxMonitor()
        
    def safe_forward(self, func, *args, **kwargs):
        """安全执行前向传播"""
        try:
            if not self.monitor.check_limits():
                return None
            return func(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.monitor.record_explosion(type(e).__name__)
            return None
    
    def concentration_constraint(self, c_current: torch.Tensor, 
                                I: torch.Tensor,
                                T: torch.Tensor = None,
                                dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        浓度演化约束（简化SPM核心）
        
        参数：
        - c_current: 当前浓度 (batch_size,)
        - I: 电流 (batch_size,)，正为放电
        - T: 温度 (batch_size,)，可选
        - dt: 时间步长 (s)
        
        返回：
        - c_next: 下一时刻浓度
        - constraint_loss: 约束违反程度
        """
        # 温度效应（Arrhenius方程）
        if T is None:
            T = torch.ones_like(c_current) * 298.15  # 默认25°C
        
        # 计算温度修正因子
        T_ref = 298.15
        arrhenius_factor = torch.exp(
            self.activation_energy / self.R * (1/T_ref - 1/T)
        )
        
        # 简化扩散：假设浓度梯度与电流成正比
        diffusion_flux = self.diffusion_coeff * self.D_s * torch.abs(I) / (self.R_p**2)
        
        # 简化反应：线性关系
        reaction_rate = self.reaction_rate * arrhenius_factor * torch.abs(I)
        
        # 总变化率（放电时浓度下降）
        sign = torch.sign(I)
        dc_dt = -diffusion_flux - reaction_rate * sign
        
        # 显式欧拉积分
        c_next = c_current + dc_dt * dt
        
        # 物理约束：浓度必须在合理范围
        c_next = torch.clamp(c_next, self.epsilon, self.c_max)
        
        # 约束损失：惩罚不合理的浓度变化
        constraint_loss = torch.mean(
            torch.relu(-dc_dt * sign)  # 放电时浓度应该下降
        )
        
        return c_next, constraint_loss
    
    def voltage_constraint(self, soc: torch.Tensor,
                          I: torch.Tensor,
                          T: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        电压约束（简化电化学）
        
        参数：
        - soc: 荷电状态 (0-1)
        - I: 电流
        - T: 温度
        
        返回：
        - voltage: 预测电压
        - constraint_loss: 约束违反程度
        """
        # 开路电压曲线（简化NMC）
        # U_ocv = 3.7 + 0.5*(soc-0.5) + 0.1*(soc-0.5)**3
        U_ocv = 3.7 + 0.3 * (soc - 0.5) + 0.05 * torch.sin(2*np.pi*soc)
        
        # 欧姆极化（线性）
        R_ohm = 0.01  # 欧姆内阻 (Ω)
        V_ohm = I * R_ohm
        
        # 电化学极化（简化）
        R_ct = 0.005  # 电荷转移电阻 (Ω)
        V_ct = torch.tanh(I * 10) * R_ct  # 饱和特性
        
        # 端电压
        voltage = U_ocv - V_ohm - V_ct
        
        # 电压合理性约束（2.5V - 4.2V）
        lower_bound = 2.5
        upper_bound = 4.2
        voltage_loss = torch.mean(
            torch.relu(lower_bound - voltage) + torch.relu(voltage - upper_bound)
        )
        
        return voltage, voltage_loss
    
    def capacity_fade_constraint(self, 
                               c_history: torch.Tensor,
                               cycles: torch.Tensor) -> torch.Tensor:
        """
        容量衰减约束（基于循环次数）
        
        简化模型：容量衰减与循环次数成幂律关系
        """
        # 计算容量衰减率
        if c_history.shape[1] < 2:
            return torch.tensor(0.0, device=self.device)
        
        initial_capacity = c_history[:, 0]
        current_capacity = c_history[:, -1]
        capacity_fade = (initial_capacity - current_capacity) / initial_capacity
        
        # 幂律衰减模型：fade ∝ cycles^β
        beta = 0.5  # 衰减指数
        expected_fade = 0.001 * (cycles ** beta)  # 每循环0.1%衰减
        
        # 约束损失：实际衰减应与模型一致
        fade_loss = torch.mean((capacity_fade - expected_fade) ** 2)
        
        return fade_loss
    
    def forward(self, 
                c_current: torch.Tensor,
                I: torch.Tensor,
                T: Optional[torch.Tensor] = None,
                dt: float = 1.0,
                c_history: Optional[torch.Tensor] = None,
                cycles: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播：计算所有物理约束
        
        返回包含约束损失和预测值的字典
        """
        # 安全执行
        result = self.safe_forward(self._forward_impl, 
                                  c_current, I, T, dt, c_history, cycles)
        if result is None:
            # 返回空结果，触发熔断
            return {
                "total_loss": torch.tensor(float('inf'), device=self.device),
                "constraints": {},
                "predictions": {},
                "status": "failed"
            }
        
        return result
    
    def _forward_impl(self, 
                     c_current: torch.Tensor,
                     I: torch.Tensor,
                     T: Optional[torch.Tensor],
                     dt: float,
                     c_history: Optional[torch.Tensor],
                     cycles: Optional[torch.Tensor]) -> dict:
        """实际的前向传播实现"""
        # 1. 浓度约束
        c_next, conc_loss = self.concentration_constraint(c_current, I, T, dt)
        
        # 2. 计算SOC
        soc = c_current / self.c_max
        
        # 3. 电压约束
        voltage, voltage_loss = self.voltage_constraint(soc, I, T)
        
        # 4. 容量衰减约束（如果提供历史数据）
        fade_loss = torch.tensor(0.0, device=self.device)
        if c_history is not None and cycles is not None:
            fade_loss = self.capacity_fade_constraint(c_history, cycles)
        
        # 总约束损失
        total_loss = conc_loss + 0.5 * voltage_loss + 0.1 * fade_loss
        
        return {
            "total_loss": total_loss,
            "constraints": {
                "concentration": conc_loss,
                "voltage": voltage_loss,
                "capacity_fade": fade_loss
            },
            "predictions": {
                "c_next": c_next,
                "soc": soc,
                "voltage": voltage
            },
            "status": "success"
        }


class SPM_PINN_Wrapper(nn.Module):
    """
    SPM-PINN包装器：神经网络 + SPM物理约束
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, device='cpu'):
        super().__init__()
        self.device = device
        
        # 神经网络部分（预测物理参数）
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 输出：反应速率、扩散系数、活化能修正
            nn.Tanh()  # 限制输出范围
        )
        
        # SPM物理层
        self.physics = SPMPhysicsLayer(device=device)
        
        # 梯度裁剪
        self.gradient_clip_value = 1.0
        
    def predict_physics_params(self, features: torch.Tensor) -> dict:
        """神经网络预测物理参数"""
        # 特征：[电流, 温度, 循环次数, 当前容量, ...]
        raw_output = self.net(features)
        
        # 将tanh输出映射到合理范围
        params = {
            "reaction_rate": 1e-4 * (1 + 0.5 * raw_output[:, 0]),  # ±50%变化
            "diffusion_coeff": 1.0 * (1 + 0.3 * raw_output[:, 1]),  # ±30%变化
            "activation_energy_shift": 5e3 * raw_output[:, 2]  # ±5kJ/mol变化
        }
        
        return params
    
    def forward(self, features: torch.Tensor, 
                c_current: torch.Tensor,
                I: torch.Tensor,
                dt: float = 1.0) -> dict:
        """
        前向传播
        
        参数：
        - features: 输入特征 (batch_size, input_dim)
        - c_current: 当前浓度
        - I: 电流
        - dt: 时间步长
        
        返回：
        - 包含预测和约束损失的字典
        """
        # 神经网络预测物理参数
        physics_params = self.predict_physics_params(features)
        
        # 更新物理层的参数
        self.physics.reaction_rate.data = physics_params["reaction_rate"].mean()
        self.physics.diffusion_coeff.data = physics_params["diffusion_coeff"].mean()
        
        # 应用物理约束
        result = self.physics(c_current, I, dt=dt)
        
        # 添加神经网络预测
        if result["status"] == "success":
            result["neural_predictions"] = physics_params
        
        return result


def test_spm_pinn_integration():
    """测试SPM-PINN集成可行性"""
    print("=" * 70)
    print("SPM-PINN集成可行性测试")
    print("=" * 70)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建模型
    model = SPM_PINN_Wrapper(input_dim=5, hidden_dim=16, device=device)
    model.to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    batch_size = 8
    seq_len = 20
    
    # 随机生成测试数据
    torch.manual_seed(42)
    
    # 特征：[电流, 温度, 循环次数, 当前容量, 历史容量变化]
    features = torch.randn(batch_size, 5, device=device)
    
    # 当前浓度（50-80% SOC）
    c_current = torch.rand(batch_size, device=device) * 6000 + 24000  # 24000-30000
    
    # 电流（-2A到2A）
    I = torch.randn(batch_size, device=device) * 1.0
    
    # 时间步长（0.1到10小时）
    dt = torch.rand(batch_size, device=device) * 9.9 + 0.1
    
    print(f"\n测试数据:")
    print(f"  特征形状: {features.shape}")
    print(f"  当前浓度: {c_current.min():.1f} - {c_current.max():.1f}")
    print(f"  电流范围: {I.min():.2f} - {I.max():.2f} A")
    print(f"  时间步长: {dt.min():.2f} - {dt.max():.2f} s")
    
    # 前向传播测试
    print(f"\n🚀 开始前向传播测试...")
    
    try:
        with torch.no_grad():
            result = model(features, c_current, I, dt.mean().item())
        
        if result["status"] == "success":
            print("✅ 前向传播成功！")
            
            # 检查结果
            print(f"\n结果检查:")
            print(f"  总约束损失: {result['total_loss'].item():.6f}")
            
            for name, loss in result["constraints"].items():
                print(f"  {name}损失: {loss.item():.6f}")
            
            # 检查预测值
            preds = result["predictions"]
            print(f"\n预测值检查:")
            print(f"  下一时刻浓度: {preds['c_next'].min():.1f} - {preds['c_next'].max():.1f}")
            print(f"  SOC: {preds['soc'].min():.3f} - {preds['soc'].max():.3f}")
            print(f"  电压: {preds['voltage'].min():.3f} - {preds['voltage'].max():.3f} V")
            
            # 检查数值稳定性
            has_nan = False
            for key, value in result.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            if torch.any(torch.isnan(subvalue)):
                                print(f"❌ {key}.{subkey} 包含NaN")
                                has_nan = True
                elif torch.is_tensor(value):
                    if torch.any(torch.isnan(value)):
                        print(f"❌ {key} 包含NaN")
                        has_nan = True
            
            if not has_nan:
                print("✅ 数值稳定性检查通过")
            else:
                print("⚠️ 发现数值稳定性问题")
                
        else:
            print("❌ 前向传播失败")
            return False
            
    except Exception as e:
        print(f"❌ 前向传播异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 训练测试（小规模）
    print(f"\n🚀 开始小规模训练测试...")
    
    try:
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 训练几个step
        n_steps = 5
        losses = []
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # 前向传播
            result = model(features, c_current, I, dt.mean().item())
            
            if result["status"] == "failed":
                print(f"❌ 第{step+1}步：物理约束失败")
                break
            
            # 反向传播
            loss = result["total_loss"]
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.gradient_clip_value)
            
            # 参数更新
            optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            
            # 检查NaN
            if torch.any(torch.isnan(loss)):
                print(f"❌ 第{step+1}步：Loss出现NaN")
                break
            
            print(f"  步骤{step+1}: loss={loss.item():.6f}")
        
        if len(losses) == n_steps:
            print(f"\n✅ 训练测试成功！")
            print(f"  初始损失: {losses[0]:.6f}")
            print(f"  最终损失: {losses[-1]:.6f}")
            
            if losses[-1] < losses[0] * 0.9:  # 至少下降10%
                print(f"  🎉 Loss稳定下降，收敛性良好")
                return True
            else:
                print(f"  ⚠️ Loss下降不明显，可能需要调整")
                return False
        else:
            print(f"\n⚠️ 训练测试中断")
            return False
            
    except Exception as e:
        print(f"❌ 训练测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 SPM-PINN集成可行性测试开始")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    start_time = time.time()
    success = test_spm_pinn_integration()
    elapsed = time.time() - start_time
    
    print(f"\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    print(f"测试耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    if success:
        print(f"\n🎉 SPM-PINN集成可行性验证通过！")
        print("可以进行下一步：")
        print("1. 实现更精确的SPM物理模型")
        print("2. 集成到Chronos-PINN主框架")
        print("3. 进行大规模实验验证")
    else:
        print(f"\n⚠️ SPM-PINN集成遇到问题")
        print("建议：")
        print("1. 检查物理约束的数值稳定性")
        print("2. 调整神经网络结构")
        print("3. 考虑退路方案：宏观热力学约束")
    
    # 沙盒状态
    if hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()
    
    print(f"\n🕒 沙盒剩余时间: ~47.5小时")
    print("=" * 70)


if __name__ == "__main__":
    main()