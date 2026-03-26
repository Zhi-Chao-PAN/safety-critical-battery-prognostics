import sys
import torch

print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU设备:", torch.cuda.get_device_name(0))
    print("CUDA版本:", torch.version.cuda)