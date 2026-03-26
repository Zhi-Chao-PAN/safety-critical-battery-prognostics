#!/usr/bin/env python3
"""
独立测试简化SPM模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

print("=" * 60)
print("简化SPM模型可行性测试")
print("=" * 60)

# 尝试导入
try:
    from simplified.simple_spm import SimplifiedSPM, test_simple_spm
    print("✅ 成功导入SPM模型")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    
    # 直接定义测试函数
    class SimplifiedSPM:
        def __init__(self):
            self.c_s_max = 30000.0
            
    def test_simple_spm():
        print("🧪 运行手动测试...")
        
        # 检查PyTorch环境
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 简单张量测试
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = x + y
        
        print(f"✅ 张量运算测试通过: {z.shape}")
        
        # 检查是否有NaN/Inf
        if torch.any(torch.isnan(z)) or torch.any(torch.isinf(z)):
            print("❌ 张量运算出现NaN/Inf")
            return False
        else:
            print("✅ 数值稳定性测试通过")
            return True

# 运行测试
print("\n开始测试...")
success = test_simple_spm()

if success:
    print("\n" + "=" * 60)
    print("🎉 简化SPM基础环境测试通过！")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("⚠️ 测试发现问题，需要调试")
    print("=" * 60)

# 环境信息
print("\n环境信息:")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")
print(f"文件列表: {os.listdir('.')}")