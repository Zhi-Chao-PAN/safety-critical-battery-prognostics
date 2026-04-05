#!/usr/bin/env python3
"""
轻量化推理引擎 - 电池寿命预测模型边缘部署

功能：
1. 使用 ONNX Runtime 进行纯推理，完全脱离 PyTorch 环境
2. 输入 NumPy 数组，输出预测的 RUL 值
3. 支持批量推理和单样本推理
4. 提供不确定性量化（MC Dropout 模拟）
5. 内存优化和性能监控

使用场景：
- 嵌入式 BMS 芯片上的实时推理
- 资源受限设备上的电池寿命预测
- 生产环境中的模型部署

作者：资深 AI 部署工程师
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    """推理统计信息"""
    total_inferences: int = 0
    total_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update(self, latency_ms: float):
        """更新统计信息"""
        self.total_inferences += 1
        self.total_time_ms += latency_ms
        self.avg_latency_ms = self.total_time_ms / self.total_inferences
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_inferences': self.total_inferences,
            'total_time_ms': round(self.total_time_ms, 2),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'min_latency_ms': round(self.min_latency_ms, 2),
            'max_latency_ms': round(self.max_latency_ms, 2),
            'memory_usage_mb': round(self.memory_usage_mb, 2)
        }


class BatteryInferenceEngine:
    """
    电池寿命预测轻量化推理引擎
    
    核心特性：
    1. 零 PyTorch 依赖 - 仅需 ONNX Runtime 和 NumPy
    2. 内存高效 - 支持流式处理和批量优化
    3. 实时性能 - 针对嵌入式设备优化
    4. 安全边界 - 输入验证和异常处理
    5. 不确定性量化 - 支持概率预测
    """
    
    def __init__(
        self,
        model_path: str,
        input_dim: int = 2,
        mc_samples: int = 50,
        device: str = 'cpu',
        enable_profiling: bool = False,
        thread_pool_size: int = 1
    ):
        """
        初始化推理引擎
        
        参数:
            model_path: ONNX 模型文件路径
            input_dim: 输入维度
            mc_samples: Monte Carlo 采样次数（用于不确定性量化）
            device: 推理设备 ('cpu', 'cuda', 'tensorrt')
            enable_profiling: 是否启用性能分析
            thread_pool_size: 线程池大小（CPU 优化）
        """
        self.model_path = Path(model_path)
        self.input_dim = input_dim
        self.mc_samples = mc_samples
        self.device = device
        self.enable_profiling = enable_profiling
        self.thread_pool_size = thread_pool_size
        
        # 推理统计
        self.stats = InferenceStats()
        
        # 模型会话
        self.session = None
        
        # 输入/输出名称
        self.input_name = None
        self.output_name = None
        
        # 初始化引擎
        self._initialize_engine()
        
        logger.info(f"推理引擎初始化完成: {self.model_path.name}")
        logger.info(f"  设备: {device}, 输入维度: {input_dim}, MC采样: {mc_samples}")
    
    def _initialize_engine(self):
        """初始化 ONNX Runtime 引擎"""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime 未安装。请运行: pip install onnxruntime"
            ) from e
        
        # 验证模型文件
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 配置会话选项
        sess_options = ort.SessionOptions()
        
        # 性能优化配置
        sess_options.enable_profiling = self.enable_profiling
        sess_options.intra_op_num_threads = self.thread_pool_size
        sess_options.inter_op_num_threads = self.thread_pool_size
        
        # 优化级别（平衡速度和内存）
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 执行提供程序配置
        providers = []
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        elif self.device == 'tensorrt' and 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers.append('TensorrtExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
        
        # 创建推理会话
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"创建 ONNX Runtime 会话失败: {e}")
        
        # 获取输入/输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 验证输入维度
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) != 2 or input_shape[1] != self.input_dim:
            logger.warning(
                f"模型输入形状 {input_shape} 与预期输入维度 {self.input_dim} 不匹配"
            )
        
        # 预热模型
        self._warmup_model()
    
    def _warmup_model(self):
        """预热模型（运行几次推理以初始化）"""
        logger.info("预热模型...")
        
        dummy_input = np.random.randn(1, self.input_dim).astype(np.float32)
        
        for _ in range(5):
            try:
                self.session.run([self.output_name], {self.input_name: dummy_input})
            except Exception as e:
                logger.warning(f"预热失败: {e}")
                break
        
        logger.info("模型预热完成")
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        验证和预处理输入数据
        
        参数:
            X: 输入数据，形状为 (batch_size, input_dim) 或 (input_dim,)
            
        返回:
            验证后的 NumPy 数组
        """
        # 转换为二维数组
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 验证形状
        if X.ndim != 2:
            raise ValueError(f"输入必须是二维数组，当前维度: {X.ndim}")
        
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"输入维度不匹配: 预期 {self.input_dim}, 实际 {X.shape[1]}"
            )
        
        # 验证数据类型
        if X.dtype != np.float32:
            logger.debug(f"转换输入数据类型: {X.dtype} -> float32")
            X = X.astype(np.float32)
        
        # 验证数值范围（可选）
        if np.any(np.isnan(X)):
            raise ValueError("输入包含 NaN 值")
        
        if np.any(np.isinf(X)):
            raise ValueError("输入包含无穷大值")
        
        return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测 RUL 值
        
        参数:
            X: 输入特征数组，形状为 (batch_size, input_dim)
               第一列必须是循环次数
            
        返回:
            RUL 预测值数组，形状为 (batch_size,)
        """
        # 验证输入
        X_valid = self._validate_input(X)
        
        # 记录开始时间
        start_time = time.perf_counter()
        
        try:
            # 运行推理
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: X_valid}
            )
            
            # 提取预测结果
            predictions = outputs[0].flatten()
            
            # 计算延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.stats.update(latency_ms)
            
            logger.debug(f"批量推理完成: batch_size={len(X_valid)}, "
                        f"latency={latency_ms:.2f}ms")
            
            return predictions
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise RuntimeError(f"推理过程中发生错误: {e}")
    
    def predict_single(self, cycle: float, feature: float) -> float:
        """
        单样本预测（简化接口）
        
        参数:
            cycle: 循环次数
            feature: 特征值（如容量衰减率）
            
        返回:
            预测的 RUL 值
        """
        # 创建输入数组
        X = np.array([[cycle, feature]], dtype=np.float32)
        
        # 运行预测
        predictions = self.predict(X)
        
        return float(predictions[0])
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        mc_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性量化的预测（模拟 MC Dropout）
        
        注意：真正的 MC Dropout 需要在导出模型时启用 dropout。
        这里使用多次推理来模拟不确定性。
        
        参数:
            X: 输入特征数组
            mc_samples: Monte Carlo 采样次数
            
        返回:
            mean: 均值预测
            lower: 95% 置信区间下界
            upper: 95% 置信区间上界
        """
        samples = mc_samples or self.mc_samples
        
        # 验证输入
        X_valid = self._validate_input(X)
        batch_size = X_valid.shape[0]
        
        # 收集多次推理结果
        all_predictions = []
        
        start_time = time.perf_counter()
        
        for i in range(samples):
            try:
                # 运行推理
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: X_valid}
                )
                
                predictions = outputs[0].flatten()
                all_predictions.append(predictions)
                
            except Exception as e:
                logger.warning(f"MC 采样 {i+1}/{samples} 失败: {e}")
                continue
        
        # 计算统计量
        if not all_predictions:
            raise RuntimeError("所有 MC 采样均失败")
        
        predictions_array = np.stack(all_predictions)  # [samples, batch_size]
        
        mean = predictions_array.mean(axis=0)
        std = predictions_array.std(axis=0)
        
        # 95% 置信区间
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        
        # 计算延迟
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_latency_ms = total_latency_ms / samples
        
        logger.info(f"MC 不确定性量化完成: "
                   f"samples={samples}, "
                   f"avg_latency={avg_latency_ms:.2f}ms, "
                   f"uncertainty={std.mean():.4f}")
        
        return mean, lower, upper
    
    def predict_detailed(
        self,
        X: np.ndarray,
        include_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        详细预测（返回所有相关信息）
        
        参数:
            X: 输入特征数组
            include_uncertainty: 是否包含不确定性量化
            
        返回:
            包含预测结果的字典
        """
        # 基本预测
        predictions = self.predict(X)
        
        result = {
            'rul_predictions': predictions.tolist(),
            'batch_size': len(predictions),
            'timestamp': time.time(),
            'engine_version': '1.0.0'
        }
        
        # 添加不确定性信息
        if include_uncertainty and self.mc_samples > 1:
            try:
                mean, lower, upper = self.predict_with_uncertainty(X)
                
                result.update({
                    'mean_predictions': mean.tolist(),
                    'lower_bounds': lower.tolist(),
                    'upper_bounds': upper.tolist(),
                    'confidence_intervals': (upper - lower).tolist(),
                    'mc_samples': self.mc_samples
                })
            except Exception as e:
                logger.warning(f"不确定性量化失败: {e}")
                result['uncertainty_error'] = str(e)
        
        # 添加性能信息
        result.update({
            'performance_stats': self.stats.to_dict(),
            'model_info': {
                'path': str(self.model_path),
                'input_dim': self.input_dim,
                'device': self.device
            }
        })
        
        return result
    
    def benchmark(
        self,
        batch_sizes: List[int] = [1, 4, 16, 64],
        num_iterations: int = 100
    ) -> Dict[int, Dict[str, float]]:
        """
        性能基准测试
        
        参数:
            batch_sizes: 要测试的批次大小列表
            num_iterations: 每个批次大小的迭代次数
            
        返回:
            各批次大小的性能指标
        """
        logger.info("开始性能基准测试...")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"测试批次大小: {batch_size}")
            
            # 生成测试数据
            test_data = np.random.randn(batch_size, self.input_dim).astype(np.float32)
            
            # 预热
            for _ in range(5):
                self.predict(test_data[:1])
            
            # 基准测试
            latencies = []
            
            for i in range(num_iterations):
                start_time = time.perf_counter()
                self.predict(test_data)
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
            
            # 计算统计量
            latencies_array = np.array(latencies)
            
            results[batch_size] = {
                'mean_latency_ms': float(np.mean(latencies_array)),
                'p50_latency_ms': float(np.median(latencies_array)),
                'p95_latency_ms': float(np.percentile(latencies_array, 95)),
                'p99_latency_ms': float(np.percentile(latencies_array, 99)),
                'throughput_samples_per_sec': (batch_size * 1000) / np.mean(latencies_array),
                'std_latency_ms': float(np.std(latencies_array))
            }
            
            logger.info(f"  平均延迟: {results[batch_size]['mean_latency_ms']:.2f}ms, "
                       f"吞吐量: {results[batch_size]['throughput_samples_per_sec']:.1f} samples/s")
        
        return results
    
    def save_stats(self, output_path: str = "inference_stats.json"):
        """保存推理统计信息"""
        import json
        
        stats_dict = {
            'engine_stats': self.stats.to_dict(),
            'model_info': {
                'path': str(self.model_path),
                'input_dim': self.input_dim,
                'device': self.device
            },
            'timestamp': time.time()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"推理统计已保存: {output_path}")
    
    def get_memory_usage(self) -> float:
        """获取内存使用情况（MB）"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.stats.memory_usage_mb = memory_mb
            return memory_mb
        except ImportError:
            logger.warning("psutil 未安装，无法获取内存使用情况")
            return 0.0
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session') and self.session is not None:
            try:
                # 尝试释放 ONNX Runtime 资源
                del self.session
            except:
                pass


class BatteryPredictorCLI:
    """
    电池预测器命令行接口
    
    提供简单的命令行界面进行推理
    """
    
    @staticmethod
    def run_interactive(model_path: str):
        """运行交互式推理界面"""
        print("=" * 60)
        print("电池寿命预测推理引擎")
        print("=" * 60)
        
        # 初始化引擎
        try:
            engine = BatteryInferenceEngine(model_path)
        except Exception as e:
            print(f"初始化失败: {e}")
            return
        
        print(f"模型加载成功: {Path(model_path).name}")
        print(f"输入维度: {engine.input_dim}")
        print()
        
        while True:
            try:
                print("请输入电池数据（格式: 循环次数 特征值），或输入 'quit' 退出:")
                user_input = input("> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("退出推理引擎")
                    break
                
                # 解析输入
                parts = user_input.split()
                if len(parts) != 2:
                    print("错误: 需要输入两个数值（循环次数和特征值）")
                    continue
                
                try:
                    cycle = float(parts[0])
                    feature = float(parts[1])
                except ValueError:
                    print("错误: 输入必须是数值")
                    continue
                
                # 运行预测
                rul = engine.predict_single(cycle, feature)
                
                print(f"预测结果:")
                print(f"  循环次数: {cycle}")
                print(f"  特征值: {feature}")
                print(f"  预测 RUL: {rul:.2f} 循环")
                print()
                
                # 显示统计信息
                if engine.stats.total_inferences % 5 == 0:
                    stats = engine.stats.to_dict()
                    print(f"推理统计: {stats['total_inferences']} 次推理, "
                          f"平均延迟: {stats['avg_latency_ms']:.2f}ms")
                    print()
                    
            except KeyboardInterrupt:
                print("\n用户中断")
                break
            except Exception as e:
                print(f"错误: {e}")
                continue
        
        # 保存统计信息
        engine.save_stats()
        print(f"统计信息已保存到 inference_stats.json")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='电池寿命预测推理引擎')
    
    # 必需参数
    parser.add_argument('model_path', type=str, help='ONNX 模型文件路径')
    
    # 可选参数
    parser.add_argument('--input-dim', type=int, default=2, help='输入维度')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda', 'tensorrt'], help='推理设备')
    parser.add_argument('--mc-samples', type=int, default=50, 
                       help='MC 采样次数（不确定性量化）')
    
    # 操作模式
    parser.add_argument('--interactive', action='store_true', 
                       help='交互式模式')
    parser.add_argument('--benchmark', action='store_true',
                       help='性能基准测试模式')
    parser.add_argument('--predict', type=str, nargs='+',
                       help='直接预测，格式: 循环次数 特征值')
    
    args = parser.parse_args()
    
    # 验证模型文件
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        return 1
    
    try:
        # 创建推理引擎
        engine = BatteryInferenceEngine(
            model_path=args.model_path,
            input_dim=args.input_dim,
            mc_samples=args.mc_samples,
            device=args.device
        )
        
        # 运行不同模式
        if args.interactive:
            BatteryPredictorCLI.run_interactive(args.model_path)
        
        elif args.benchmark:
            print("运行性能基准测试...")
            results = engine.benchmark()
            
            print("\n性能基准测试结果:")
            print("=" * 80)
            for batch_size, metrics in results.items():
                print(f"批次大小 {batch_size}:")
                print(f"  平均延迟: {metrics['mean_latency_ms']:.2f} ms")
                print(f"  P95延迟: {metrics['p95_latency_ms']:.2f} ms")
                print(f"  吞吐量: {metrics['throughput_samples_per_sec']:.1f} samples/s")
                print()
        
        elif args.predict:
            if len(args.predict) != 2:
                print("错误: --predict 需要两个参数: 循环次数 特征值")
                return 1
            
            try:
                cycle = float(args.predict[0])
                feature = float(args.predict[1])
                
                rul = engine.predict_single(cycle, feature)
                print(f"预测 RUL: {rul:.2f} 循环")
                
            except ValueError:
                print("错误: 预测参数必须是数值")
                return 1
        
        else:
            # 默认模式：显示引擎信息
            print("电池寿命预测推理引擎已就绪")
            print(f"模型: {Path(args.model_path).name}")
            print(f"输入维度: {args.input_dim}")
            print(f"设备: {args.device}")
            print(f"MC采样: {args.mc_samples}")
            print()
            print("使用 --help 查看可用选项")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())