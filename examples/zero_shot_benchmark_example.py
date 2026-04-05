"""
零样本跨数据集评测示例 (Zero-Shot Cross-Dataset Benchmark Example)

本示例展示如何使用 ZeroShotBenchmarkRunner 进行跨数据集零样本泛化评测。
主要场景：在 NASA 数据集上训练模型，直接在 CALCE 数据集上测试（无微调）。

作者: AI Assistant
日期: 2025
"""

import logging
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.zero_shot_benchmark import (
    ZeroShotBenchmarkRunner,
    ZeroShotResult,
)
from src.data.unified_loader import UnifiedDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_1_basic_zero_shot():
    """
    示例 1: 基础零样本评测

    在 NASA 数据集上训练，直接在 CALCE 上测试。
    这是最经典的零样本泛化场景。
    """
    logger.info("\n" + "="*70)
    logger.info("示例 1: 基础零样本评测 (NASA → CALCE)")
    logger.info("="*70)

    # 创建评测器
    benchmark = ZeroShotBenchmarkRunner(
        results_dir="results/example_1_nasa_to_calce",
        random_seed=42,
    )

    # 首先加载数据以推断特征维度
    data_loader = UnifiedDataLoader()
    sample_df = data_loader.load_nasa()

    # 推断特征列
    exclude_cols = ["battery_id", "dataset_source", "chemistry", "cycle", "rul", "RUL", "raw_"]
    features = [
        c for c in sample_df.columns
        if not any(ex in c.lower() for ex in exclude_cols)
        and sample_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    logger.info(f"推断的特征维度: {len(features)}")
    logger.info(f"特征列表: {features}")

    # 创建模型（这里使用简化的基线模型）
    # 实际使用时，替换为: from src.models.pinn_model import PINNModel
    from src.models.lstm_model import LSTMModel

    model = LSTMModel(
        input_dim=len(features),
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    )

    logger.info(f"创建模型: {model.name}")

    # 运行零样本评测
    result = benchmark.run_zero_shot(
        model=model,
        model_name="LSTM_NASA_to_CALCE",
        train_dataset="nasa",
        test_dataset="calce",
        features=features,
        target="rul",
        save_model=True,
    )

    logger.info("\n评测结果:")
    logger.info(f"  RMSE: {result.rmse:.4f}")
    logger.info(f"  MAE:  {result.mae:.4f}")
    logger.info(f"  PICP: {result.picp:.4f}")
    logger.info(f"  CRPS: {result.crps:.4f}")

    # 生成报告和可视化
    report_path = benchmark.generate_markdown_report(
        title="零样本评测示例: NASA → CALCE",
    )
    plot_paths = benchmark.generate_comparison_plots()

    logger.info(f"\n报告已保存: {report_path}")
    logger.info(f"可视化图表: {len(plot_paths)} 张")

    return result


def example_2_cross_dataset_matrix():
    """
    示例 2: 完整跨数据集矩阵评测

    评测所有训练集和测试集的组合，生成完整的零样本泛化矩阵。
    """
    logger.info("\n" + "="*70)
    logger.info("示例 2: 完整跨数据集矩阵评测")
    logger.info("="*70)

    # 使用脚本运行完整矩阵
    from scripts.run_zero_shot_benchmark import run_full_matrix_evaluation

    results = run_full_matrix_evaluation(
        model_name="lstm",
        datasets=["nasa", "calce"],
        results_dir="results/example_2_matrix",
    )

    logger.info("\n完整矩阵评测完成!")
    logger.info(f"结果数量: {len(results['results'])}")

    return results


def example_3_custom_model_integration():
    """
    示例 3: 自定义模型集成

    展示如何将自定义 PINN 模型集成到评测流水线中。
    """
    logger.info("\n" + "="*70)
    logger.info("示例 3: 自定义 PINN 模型集成")
    logger.info("="*70)

    # 假设用户有自己的 PINN 模型类
    # from src.models.pinn_model import PINNModel

    # 示例: 如何包装现有模型以适配评测流水线
    class MyCustomPINN:
        """用户自定义 PINN 模型示例"""

        name = "MyCustomPINN"

        def __init__(self, input_dim: int):
            self.input_dim = input_dim
            # 初始化模型参数...

        def fit(self, X, y, **kwargs):
            """训练模型"""
            logger.info(f"Training on {len(X)} samples...")
            # 训练逻辑...
            return self

        def predict(self, X, **kwargs):
            """预测并返回均值和置信区间"""
            logger.info(f"Predicting on {len(X)} samples...")
            # 预测逻辑...
            # 返回: (mean, lower, upper)
            mean = np.random.randn(len(X))  # 示例
            lower = mean - 0.1
            upper = mean + 0.1
            return mean, lower, upper

        def save(self, path):
            """保存模型"""
            logger.info(f"Saving model to {path}")

        def load(self, path):
            """加载模型"""
            logger.info(f"Loading model from {path}")
            return self

    # 使用自定义模型运行评测
    benchmark = ZeroShotBenchmarkRunner(
        results_dir="results/example_3_custom",
        random_seed=42,
    )

    # 创建模型实例
    custom_model = MyCustomPINN(input_dim=8)

    # 运行评测
    # result = benchmark.run_zero_shot(
    #     model=custom_model,
    #     model_name="MyCustomPINN",
    #     train_dataset="nasa",
    #     test_dataset="calce",
    # )

    logger.info("\n自定义模型集成示例完成!")
    logger.info("请将 MyCustomPINN 替换为您的实际模型类")

    return custom_model


def main():
    """主函数：运行所有示例"""
    logger.info("\n" + "="*70)
    logger.info("零样本跨数据集评测示例")
    logger.info("Zero-Shot Cross-Dataset Benchmark Examples")
    logger.info("="*70)

    # 选择要运行的示例
    examples = {
        "1": ("基础零样本评测", example_1_basic_zero_shot),
        "2": ("完整跨数据集矩阵", example_2_cross_dataset_matrix),
        "3": ("自定义模型集成", example_3_custom_model_integration),
    }

    logger.info("\n可用示例:")
    for key, (name, _) in examples.items():
        logger.info(f"  {key}. {name}")

    # 默认运行示例 1
    choice = input("\n选择要运行的示例 (1-3, 默认 1): ").strip() or "1"

    if choice in examples:
        name, func = examples[choice]
        logger.info(f"\n运行: {name}")
        try:
            result = func()
            logger.info(f"\n✅ {name} 完成!")
            return result
        except Exception as e:
            logger.error(f"\n❌ {name} 失败: {e}")
            raise
    else:
        logger.error(f"无效选择: {choice}")
        return None


if __name__ == "__main__":
    main()