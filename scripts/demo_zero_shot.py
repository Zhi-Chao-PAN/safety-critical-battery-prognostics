"""
零样本跨数据集评测流水线演示 (Zero-Shot Benchmark Demo)

展示如何使用 ZeroShotBenchmarkRunner 进行业界标杆级别的零样本泛化评测。

运行方式:
    python demo_zero_shot.py

作者: AI Assistant
"""

import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("\n" + "="*70)
    logger.info("零样本跨数据集评测流水线 (Zero-Shot Benchmark)")
    logger.info("="*70)
    
    # 导入核心类
    try:
        from src.evaluation import ZeroShotBenchmarkRunner, ZeroShotResult
        logger.info("✅ 核心类导入成功")
    except ImportError as e:
        logger.error(f"❌ 导入失败: {e}")
        logger.error("请确保在项目根目录运行此脚本")
        return 1
    
    # 显示功能概览
    logger.info("\n📋 功能概览:")
    logger.info("  1. 零样本泛化测试 (Dataset A → Dataset B, 无微调)")
    logger.info("  2. 支持多数据集: NASA PCoE, CALCE CS2, Oxford, MIT")
    logger.info("  3. 全面评测指标: RMSE, MAE, PICP, CRPS, Coverage")
    logger.info("  4. 自动化 Markdown 报告生成")
    logger.info("  5. 丰富的可视化图表 (热力图, 对比图, 箱线图)")
    
    # 显示使用方式
    logger.info("\n💻 使用方式:")
    logger.info("\n  Python API:")
    logger.info("    from src.evaluation import ZeroShotBenchmarkRunner")
    logger.info("    benchmark = ZeroShotBenchmarkRunner()")
    logger.info("    result = benchmark.run_zero_shot(model, 'nasa', 'calce')")
    
    logger.info("\n  命令行:")
    logger.info("    # 单组评测")
    logger.info("    python scripts/run_zero_shot_benchmark.py \\")
    logger.info("        --model pinn --train nasa --test calce")
    logger.info("")
    logger.info("    # 完整矩阵评测")
    logger.info("    python scripts/run_zero_shot_benchmark.py \\")
    logger.info("        --model pinn --run-full-matrix \\")
    logger.info("        --datasets nasa calce")
    
    # 显示文件结构
    logger.info("\n📁 生成的文件结构:")
    logger.info("  results/zero_shot_benchmark/")
    logger.info("  ├── zero_shot_benchmark_report.md    # Markdown 报告")
    logger.info("  ├── zero_shot_results.json           # JSON 结果")
    logger.info("  └── figures/")
    logger.info("      ├── zero_shot_heatmap_rmse.png   # RMSE 热力图")
    logger.info("      ├── zero_shot_heatmap_picp.png   # PICP 热力图")
    logger.info("      ├── zero_shot_comparison.png     # 对比图")
    logger.info("      └── zero_shot_boxplot.png        # 箱线图")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成! 详细使用方法请参考:")
    logger.info("  - examples/zero_shot_benchmark_example.py")
    logger.info("  - scripts/run_zero_shot_benchmark.py --help")
    logger.info("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())