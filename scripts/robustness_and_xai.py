"""
工业级QA压力测试与可解释性分析脚本
Robustness Testing & XAI Analysis for Battery RUL Prediction

极端工况测试：
- 高斯噪声注入：20%/50%/80% 信噪比
- 传感器掉线模拟：随机NaN注入
- 物理约束开关对比实验：验证物理约束的鲁棒性提升
- SHAP可解释性：分析物理特征对预测的贡献度
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import shap

# 项目内部导入
from src.models.pinn_model import PINNModel
from src.data.unified_loader import UnifiedDataLoader
from src.features.extractor import FeatureExtractor
from src.physics.constraints import MonotonicityConstraint
from src.utils.metrics import compute_rmse, compute_mae

# 配置参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_LEVELS = [0.2, 0.5, 0.8]  # 噪声强度：0.2=20%噪声
NAN_RATIOS = [0.2, 0.5, 0.8]    # NaN比例：模拟传感器掉线
FEATURE_NAMES = [
    "cycle_count", "discharge_capacity", "charge_capacity",
    "internal_resistance", "max_voltage", "min_voltage",
    "mean_temperature", "max_temperature", "ic_peak_height",
    "ic_peak_position", "dv_peak_height", "max_concentration_gradient"
]

def inject_gaussian_noise(X: np.ndarray, noise_level: float) -> np.ndarray:
    """
    注入高斯噪声模拟传感器噪声
    
    Args:
        X: 原始特征矩阵 [n_samples, n_features]
        noise_level: 噪声强度 (0-1)，表示噪声标准差为特征标准差的比例
    
    Returns:
        加入噪声后的特征矩阵
    """
    X_noisy = X.copy()
    for i in range(X.shape[1]):
        # 跳过cycle_count特征，不添加噪声
        if i == 0:
            continue
        std = np.std(X[:, i])
        noise = np.random.normal(0, noise_level * std, size=X.shape[0])
        X_noisy[:, i] += noise
    return X_noisy

def inject_nan_values(X: np.ndarray, nan_ratio: float) -> np.ndarray:
    """
    随机注入NaN值模拟传感器掉线
    
    Args:
        X: 原始特征矩阵
        nan_ratio: NaN值比例 (0-1)
    
    Returns:
        包含NaN的特征矩阵
    """
    X_nan = X.copy()
    # 跳过cycle_count特征，不设置NaN
    mask = np.random.choice([True, False], size=X.shape, p=[nan_ratio, 1-nan_ratio])
    mask[:, 0] = False  # 第一列是cycle_count，不添加NaN
    X_nan[mask] = np.nan
    
    # 简单插值填充NaN（工业级场景可替换为更复杂的填充策略）
    for i in range(X_nan.shape[1]):
        nans = np.isnan(X_nan[:, i])
        if np.any(nans):
            X_nan[nans, i] = np.nanmean(X_nan[:, i])
    return X_nan

def compute_physical_violation_rate(predictions: np.ndarray) -> float:
    """
    计算物理违规率：非单调递减的次数比例（容量/RUL应该随循环单调递减）
    
    Args:
        predictions: 预测值序列，按cycle排序
    
    Returns:
        违规率 (0-1)
    """
    if len(predictions) < 2:
        return 0.0
    diffs = np.diff(predictions)
    # 正差值表示预测值上升，违反单调递减约束
    violations = np.sum(diffs > 1e-6)  # 小阈值避免浮点误差
    return violations / len(diffs)

def run_robustness_experiment(
    model_with_physics: PINNModel,
    model_data_only: PINNModel,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    运行鲁棒性对比实验，对比物理约束开启/关闭的性能
    
    Args:
        model_with_physics: 开启物理约束的模型
        model_data_only: 纯数据驱动模型（关闭物理约束）
        X_test: 测试集特征
        y_test: 测试集标签
    
    Returns:
        实验结果DataFrame
    """
    results = []
    
    # 基准测试（无噪声）
    y_pred_physics, _, _ = model_with_physics.predict(X_test)
    y_pred_data, _, _ = model_data_only.predict(X_test)
    
    results.append({
        "noise_level": 0.0,
        "nan_ratio": 0.0,
        "model_type": "physics_constrained",
        "rmse": compute_rmse(y_test, y_pred_physics),
        "violation_rate": compute_physical_violation_rate(y_pred_physics)
    })
    results.append({
        "noise_level": 0.0,
        "nan_ratio": 0.0,
        "model_type": "data_only",
        "rmse": compute_rmse(y_test, y_pred_data),
        "violation_rate": compute_physical_violation_rate(y_pred_data)
    })
    
    # 高斯噪声测试
    for noise_level in tqdm(NOISE_LEVELS, desc="Gaussian Noise Testing"):
        X_noisy = inject_gaussian_noise(X_test, noise_level)
        
        y_pred_physics, _, _ = model_with_physics.predict(X_noisy)
        y_pred_data, _, _ = model_data_only.predict(X_noisy)
        
        results.append({
            "noise_level": noise_level,
            "nan_ratio": 0.0,
            "model_type": "physics_constrained",
            "rmse": compute_rmse(y_test, y_pred_physics),
            "violation_rate": compute_physical_violation_rate(y_pred_physics)
        })
        results.append({
            "noise_level": noise_level,
            "nan_ratio": 0.0,
            "model_type": "data_only",
            "rmse": compute_rmse(y_test, y_pred_data),
            "violation_rate": compute_physical_violation_rate(y_pred_data)
        })
    
    # NaN注入测试
    for nan_ratio in tqdm(NAN_RATIOS, desc="NaN Injection Testing"):
        X_nan = inject_nan_values(X_test, nan_ratio)
        
        y_pred_physics, _, _ = model_with_physics.predict(X_nan)
        y_pred_data, _, _ = model_data_only.predict(X_nan)
        
        results.append({
            "noise_level": 0.0,
            "nan_ratio": nan_ratio,
            "model_type": "physics_constrained",
            "rmse": compute_rmse(y_test, y_pred_physics),
            "violation_rate": compute_physical_violation_rate(y_pred_physics)
        })
        results.append({
            "noise_level": 0.0,
            "nan_ratio": nan_ratio,
            "model_type": "data_only",
            "rmse": compute_rmse(y_test, y_pred_data),
            "violation_rate": compute_physical_violation_rate(y_pred_data)
        })
    
    # 混合噪声测试
    for noise_level, nan_ratio in tqdm(zip(NOISE_LEVELS, NAN_RATIOS), desc="Mixed Noise Testing", total=len(NOISE_LEVELS)):
        X_mixed = inject_nan_values(inject_gaussian_noise(X_test, noise_level), nan_ratio)
        
        y_pred_physics, _, _ = model_with_physics.predict(X_mixed)
        y_pred_data, _, _ = model_data_only.predict(X_mixed)
        
        results.append({
            "noise_level": noise_level,
            "nan_ratio": nan_ratio,
            "model_type": "physics_constrained",
            "rmse": compute_rmse(y_test, y_pred_physics),
            "violation_rate": compute_physical_violation_rate(y_pred_physics)
        })
        results.append({
            "noise_level": noise_level,
            "nan_ratio": nan_ratio,
            "model_type": "data_only",
            "rmse": compute_rmse(y_test, y_pred_data),
            "violation_rate": compute_physical_violation_rate(y_pred_data)
        })
    
    return pd.DataFrame(results)

def run_shap_analysis(model: PINNModel, X_train: np.ndarray, X_test: np.ndarray) -> None:
    """
    运行SHAP可解释性分析，生成特征贡献度可视化
    
    Args:
        model: 训练好的PINN模型
        X_train: 训练集特征（用于初始化SHAP解释器）
        X_test: 测试集特征（用于生成解释）
    """
    # 定义预测函数
    def predict_fn(x):
        mean, _, _ = model.predict(x)
        return mean
    
    # 初始化SHAP解释器
    explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train, 100))
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test, nsamples=100)
    
    # 生成Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False
    )
    plt.title("Feature Importance (SHAP Values)")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png", dpi=300, bbox_inches="tight")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    
    # 分析物理特征（最大浓度梯度）的贡献
    if "max_concentration_gradient" in FEATURE_NAMES:
        feat_idx = FEATURE_NAMES.index("max_concentration_gradient")
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx, 
            shap_values, 
            X_test,
            feature_names=FEATURE_NAMES,
            show=False
        )
        plt.title("Max Concentration Gradient SHAP Dependence Plot")
        plt.tight_layout()
        plt.savefig("shap_concentration_gradient_dependence.png", dpi=300, bbox_inches="tight")
    
    # 保存SHAP结果
    np.save("shap_values.npy", shap_values)
    print("SHAP分析完成，结果已保存到当前目录")

def plot_experiment_results(results_df: pd.DataFrame) -> None:
    """
    可视化实验结果
    """
    # 绘制RMSE对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 高斯噪声RMSE对比
    noise_data = results_df[results_df["nan_ratio"] == 0.0]
    for model_type in ["physics_constrained", "data_only"]:
        subset = noise_data[noise_data["model_type"] == model_type]
        ax1.plot(
            subset["noise_level"], 
            subset["rmse"], 
            marker="o", 
            linewidth=2,
            label="Physics-Constrained" if model_type == "physics_constrained" else "Data-Only"
        )
    ax1.set_xlabel("Gaussian Noise Level")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs Gaussian Noise Level")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 物理违规率对比
    for model_type in ["physics_constrained", "data_only"]:
        subset = noise_data[noise_data["model_type"] == model_type]
        ax2.plot(
            subset["noise_level"], 
            subset["violation_rate"], 
            marker="s", 
            linewidth=2,
            label="Physics-Constrained" if model_type == "physics_constrained" else "Data-Only"
        )
    ax2.set_xlabel("Gaussian Noise Level")
    ax2.set_ylabel("Physical Violation Rate")
    ax2.set_title("Monotonicity Violation Rate vs Gaussian Noise Level")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("robustness_comparison.png", dpi=300, bbox_inches="tight")
    
    # 绘制NaN注入结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    nan_data = results_df[results_df["noise_level"] == 0.0]
    
    for model_type in ["physics_constrained", "data_only"]:
        subset = nan_data[nan_data["model_type"] == model_type]
        ax1.plot(
            subset["nan_ratio"], 
            subset["rmse"], 
            marker="o", 
            linewidth=2,
            label="Physics-Constrained" if model_type == "physics_constrained" else "Data-Only"
        )
    ax1.set_xlabel("NaN Ratio (Sensor Dropout)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs Sensor Dropout Ratio")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    for model_type in ["physics_constrained", "data_only"]:
        subset = nan_data[nan_data["model_type"] == model_type]
        ax2.plot(
            subset["nan_ratio"], 
            subset["violation_rate"], 
            marker="s", 
            linewidth=2,
            label="Physics-Constrained" if model_type == "physics_constrained" else "Data-Only"
        )
    ax2.set_xlabel("NaN Ratio (Sensor Dropout)")
    ax2.set_ylabel("Physical Violation Rate")
    ax2.set_title("Monotonicity Violation Rate vs Sensor Dropout Ratio")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sensor_dropout_comparison.png", dpi=300, bbox_inches="tight")
    
    print("实验结果可视化完成，图片已保存到当前目录")

def main():
    """主函数"""
    print("="*80)
    print("工业级QA极端工况压力测试与可解释性分析")
    print("="*80)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据集...")
    loader = UnifiedDataLoader(dataset_name="calce")
    X_train, y_train, X_test, y_test = loader.load_splits()
    
    # 2. 提取SPM物理特征（最大浓度梯度）
    print("\n[2/6] 提取物理特征...")
    extractor = FeatureExtractor()
    X_train = extractor.extract_features(X_train)
    X_test = extractor.extract_features(X_test)
    
    # 3. 训练两个对比模型
    print("\n[3/6] 训练模型...")
    # 开启物理约束的模型
    model_physics = PINNModel(
        input_dim=X_train.shape[1],
        lambda_physics=0.1,
        lambda_mono=0.05,
        adaptive_weighting=True,
        device=DEVICE
    )
    model_physics.fit(X_train, y_train)
    
    # 纯数据驱动模型（关闭物理约束）
    model_data = PINNModel(
        input_dim=X_train.shape[1],
        lambda_physics=0.0,  # 关闭物理约束
        lambda_mono=0.0,     # 关闭单调性约束
        adaptive_weighting=False,
        device=DEVICE
    )
    model_data.fit(X_train, y_train)
    
    # 4. 运行鲁棒性实验
    print("\n[4/6] 运行鲁棒性对比实验...")
    results_df = run_robustness_experiment(model_physics, model_data, X_test, y_test)
    results_df.to_csv("robustness_experiment_results.csv", index=False)
    print("实验结果已保存到 robustness_experiment_results.csv")
    
    # 5. 可视化实验结果
    print("\n[5/6] 生成实验结果可视化...")
    plot_experiment_results(results_df)
    
    # 6. 运行SHAP可解释性分析
    print("\n[6/6] 运行SHAP可解释性分析...")
    run_shap_analysis(model_physics, X_train, X_test)
    
    print("\n" + "="*80)
    print("所有分析完成！")
    print("="*80)
    
    # 打印关键结果对比
    print("\n关键结果摘要：")
    summary = results_df.groupby(["model_type", "noise_level", "nan_ratio"]).agg({
        "rmse": "mean",
        "violation_rate": "mean"
    }).reset_index()
    
    print("\n噪声水平80%时性能对比：")
    high_noise = summary[(summary["noise_level"] == 0.8) & (summary["nan_ratio"] == 0.0)]
    print(high_noise.to_string(index=False))
    
    print("\n传感器掉线80%时性能对比：")
    high_nan = summary[(summary["nan_ratio"] == 0.8) & (summary["noise_level"] == 0.0)]
    print(high_nan.to_string(index=False))

if __name__ == "__main__":
    main()
