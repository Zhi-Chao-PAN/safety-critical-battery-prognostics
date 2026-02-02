
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Lock random seeds for all libraries to ensure reproducibility.
    Essential for "Scientific" code.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")
