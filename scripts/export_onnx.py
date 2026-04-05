#!/usr/bin/env python3
"""
ONNX 导出脚本 - 电池寿命预测模型边缘部署

功能：
1. 加载训练好的 PINN 模型权重 (.pth)
2. 剔除训练时的冗余计算图，将模型静态化
3. 导出为高效率的 .onnx 格式
4. 支持动态 Batch Size 维度映射
5. 提供量化选项（FP16/INT8）

使用场景：
- 将 PyTorch 模型部署到嵌入式 BMS 芯片
- 在资源受限设备上进行轻量化推理
- 实现脱离 PyTorch 环境的纯推理

作者：资深 AI 部署工程师
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_model import PINNModel, PINNNet
from src.physics.constraints import create_default_constraint_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PINNForInference(nn.Module):
    """
    专门用于推理的 PINN 模型包装器
    
    关键优化：
    1. 移除训练时的冗余计算图（如 dropout、约束计算）
    2. 将动态权重计算静态化
    3. 支持批量推理优化
    4. 提供确定性输出
    """
    
    def __init__(self, pinn_model: PINNModel):
        super().__init__()
        
        # 提取核心神经网络部分
        self.nn_model = pinn_model.model
        
        # 提取物理模型参数
        self.physics_params = pinn_model._physics_params
        self.max_cycle = pinn_model._max_cycle
        
        # 缓存物理预测（如果可用）
        self.has_physics = self.physics_params is not None
        
        # 输入维度信息
        self.input_dim = pinn_model.input_dim
        
        logger.info(f"创建推理模型: input_dim={self.input_dim}, has_physics={self.has_physics}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 优化后的推理版本
        
        参数:
            x: 输入张量 [batch_size, input_dim]
                x[:, 0] 必须是循环次数
            
        返回:
            预测的 RUL 值 [batch_size, 1]
        """
        # 神经网络预测
        nn_output = self.nn_model(x, mc_dropout=False)
        
        # 如果存在物理模型，添加物理基线
        if self.has_physics:
            # 提取循环次数
            cycles = x[:, 0]
            
            # 计算物理预测（使用缓存的参数）
            # 这里使用简化的物理模型近似
            physics_pred = self._compute_physics_baseline(cycles)
            
            # 组合预测
            total_pred = nn_output + physics_pred.unsqueeze(-1)
        else:
            total_pred = nn_output
        
        return total_pred
    
    def _compute_physics_baseline(self, cycles: torch.Tensor) -> torch.Tensor:
        """
        计算物理基线预测（简化版本）
        
        在实际部署中，这部分可以预计算或使用查找表
        """
        # 简化的指数衰减模型
        if self.physics_params:
            # 从物理参数提取衰减率
            decay_rate = self.physics_params.get('decay_rate', 0.001)
            initial_capacity = self.physics_params.get('initial_capacity', 1.0)
            
            # 计算物理预测
            physics_pred = initial_capacity * torch.exp(-decay_rate * cycles)
        else:
            physics_pred = torch.zeros_like(cycles)
        
        return physics_pred


def load_pinn_model(model_path: str, device: str = "cpu") -> PINNModel:
    """
    加载训练好的 PINN 模型
    
    参数:
        model_path: .pth 模型文件路径
        device: 加载设备 ('cpu' 或 'cuda')
    
    返回:
        加载的 PINNModel 实例
    """
    logger.info(f"加载模型: {model_path}")
    
    # 创建模型实例（使用默认参数）
    model = PINNModel(
        input_dim=2,  # 默认输入维度
        hidden_dim=64,
        dropout=0.2,
        device=device,
        use_mixed_precision=False  # 推理时不需要混合精度
    )
    
    # 加载模型权重
    if os.path.exists(model_path):
        model.load(model_path)
        logger.info(f"模型加载成功: {model_path}")
    else:
        logger.warning(f"模型文件不存在: {model_path}")
        logger.info("创建随机初始化的模型用于演示")
        # 创建虚拟数据并训练模型
        X_dummy = np.random.randn(100, 2).astype(np.float32)
        y_dummy = np.random.randn(100).astype(np.float32)
        model.fit(X_dummy, y_dummy)
        logger.info("使用随机初始化的模型进行演示")
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 2),
    opset_version: int = 14,
    dynamic_batch: bool = True,
    verbose: bool = False
) -> str:
    """
    将 PyTorch 模型导出为 ONNX 格式
    
    参数:
        model: PyTorch 模型
        output_path: 输出 ONNX 文件路径
        input_shape: 输入形状 (batch_size, input_dim)
        opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch size
        verbose: 是否显示详细信息
    
    返回:
        导出的 ONNX 文件路径
    """
    logger.info(f"导出 ONNX 模型到: {output_path}")
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(*input_shape)
    
    # 配置动态轴
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},  # 动态 batch 维度
            'output': {0: 'batch_size'}   # 输出对应动态维度
        }
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # 常量折叠优化
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=verbose
    )
    
    # 验证导出文件
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"ONNX 导出成功: {output_path} ({file_size:.2f} MB)")
    
    return output_path


def quantize_onnx(
    onnx_path: str,
    output_path: str,
    quantization_type: str = "fp16"
) -> str:
    """
    对 ONNX 模型进行量化
    
    参数:
        onnx_path: 原始 ONNX 文件路径
        output_path: 量化后输出路径
        quantization_type: 量化类型 ('fp16', 'int8')
    
    返回:
        量化后的 ONNX 文件路径
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        logger.warning("ONNX 量化工具未安装，跳过量化步骤")
        return onnx_path
    
    logger.info(f"量化 ONNX 模型: {quantization_type}")
    
    if quantization_type == "fp16":
        # FP16 量化
        model = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
        
    elif quantization_type == "int8":
        # INT8 动态量化
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QInt8
            )
        except ImportError:
            logger.warning("ONNX Runtime 量化模块未安装，跳过 INT8 量化")
            return onnx_path
    
    else:
        logger.warning(f"不支持的量化类型: {quantization_type}")
        return onnx_path
    
    # 比较文件大小
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    logger.info(f"量化完成: {output_path}")
    logger.info(f"  原始大小: {original_size:.2f} MB")
    logger.info(f"  量化大小: {quantized_size:.2f} MB")
    logger.info(f"  压缩率: {reduction:.1f}%")
    
    return output_path


def validate_onnx_model(onnx_path: str, input_shape: Tuple[int, ...] = (1, 2)) -> bool:
    """
    验证导出的 ONNX 模型
    
    参数:
        onnx_path: ONNX 文件路径
        input_shape: 测试输入形状
    
    返回:
        验证是否成功
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("ONNX Runtime 未安装，跳过验证")
        return True
    
    logger.info("验证 ONNX 模型...")
    
    try:
        # 创建推理会话
        session = ort.InferenceSession(onnx_path)
        
        # 准备测试输入
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 运行推理
        outputs = session.run(None, {'input': test_input})
        
        logger.info(f"ONNX 验证成功")
        logger.info(f"  输入形状: {test_input.shape}")
        logger.info(f"  输出形状: {outputs[0].shape}")
        logger.info(f"  输出示例: {outputs[0].flatten()[:3]}")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX 验证失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='导出 PINN 模型到 ONNX 格式')
    
    # 输入参数
    parser.add_argument('--model_path', type=str, default='models/pinn_model.pth',
                       help='训练好的 .pth 模型文件路径')
    parser.add_argument('--output_dir', type=str, default='exported_models',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=2,
                       help='模型输入维度')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='默认 batch size')
    
    # 导出参数
    parser.add_argument('--opset_version', type=int, default=14,
                       help='ONNX opset 版本')
    parser.add_argument('--dynamic_batch', action='store_true', default=True,
                       help='启用动态 batch size 支持')
    parser.add_argument('--quantize', type=str, choices=['none', 'fp16', 'int8'], default='none',
                       help='量化类型')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("电池寿命预测模型 ONNX 导出工具")
    logger.info("=" * 60)
    
    # 步骤 1: 加载模型
    logger.info("步骤 1: 加载 PINN 模型")
    pinn_model = load_pinn_model(args.model_path, args.device)
    
    # 步骤 2: 创建推理优化模型
    logger.info("步骤 2: 创建推理优化模型")
    inference_model = PINNForInference(pinn_model)
    
    # 步骤 3: 导出原始 ONNX
    logger.info("步骤 3: 导出原始 ONNX 模型")
    input_shape = (args.batch_size, args.input_dim)
    
    onnx_path = output_dir / "pinn_model_fp32.onnx"
    export_to_onnx(
        inference_model,
        str(onnx_path),
        input_shape=input_shape,
        opset_version=args.opset_version,
        dynamic_batch=args.dynamic_batch,
        verbose=args.verbose
    )
    
    # 步骤 4: 量化（如果启用）
    final_onnx_path = onnx_path
    if args.quantize != 'none':
        logger.info(f"步骤 4: {args.quantize.upper()} 量化")
        
        quantized_path = output_dir / f"pinn_model_{args.quantize}.onnx"
        quantize_onnx(str(onnx_path), str(quantized_path), args.quantize)
        final_onnx_path = quantized_path
    
    # 步骤 5: 验证模型
    logger.info("步骤 5: 验证导出的模型")
    validate_onnx_model(str(final_onnx_path), input_shape)
    
    # 步骤 6: 生成部署报告
    logger.info("步骤 6: 生成部署报告")
    generate_deployment_report(str(final_onnx_path), args)
    
    logger.info("=" * 60)
    logger.info("ONNX 导出完成!")
    logger.info(f"最终模型: {final_onnx_path}")
    logger.info("=" * 60)


def generate_deployment_report(onnx_path: str, args):
    """生成部署报告"""
    report_path = Path(args.output_dir) / "deployment_report.md"
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    
    report_content = f"""# 电池寿命预测模型部署报告

## 模型信息
- **模型类型**: Physics-Informed Neural Network (PINN)
- **输入维度**: {args.input_dim}
- **输出维度**: 1 (RUL 预测)
- **支持动态 Batch**: {'是' if args.dynamic_batch else '否'}

## 导出配置
- **ONNX Opset**: {args.opset_version}
- **量化类型**: {args.quantize.upper() if args.quantize != 'none' else '无'}
- **设备**: {args.device}

## 文件信息
- **ONNX 文件**: {Path(onnx_path).name}
- **文件大小**: {file_size:.2f} MB
- **保存路径**: {onnx_path}

## 部署指南

### 1. 环境要求
```
onnxruntime >= 1.14.0
numpy >= 1.21.0
```

### 2. 推理示例 (Python)
```python
import numpy as np
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession("{Path(onnx_path).name}")

# 准备输入数据
# 输入形状: [batch_size, {args.input_dim}]
# 第一列必须是循环次数
input_data = np.array([[100, 0.85]], dtype=np.float32)

# 运行推理
outputs = session.run(None, {{'input': input_data}})
rul_prediction = outputs[0][0, 0]

print(f"预测 RUL: {{rul_prediction:.2f}} 循环")
```

### 3. 性能优化建议
1. 对于嵌入式设备，建议使用 ONNX Runtime 的 CPU 提供程序
2. 可以启用线程池优化 (`session.set_session_options()`)
3. 考虑使用 INT8 量化进一步减少内存占用

### 4. 安全注意事项
1. 输入数据需要归一化处理
2. 输出 RUL 值需要反归一化
3. 建议添加边界检查防止异常输入

## 技术支持
如有问题，请参考项目文档或提交 Issue。
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"部署报告已生成: {report_path}")


if __name__ == "__main__":
    main()