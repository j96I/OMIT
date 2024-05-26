import torch

import sys
import platform
import pandas as pd
import sklearn as sk

def setup_validate():
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch,'has_mps',False)
    device = "mps" if getattr(torch, str(has_mps) , False) else "gpu" if torch.cuda.is_available() else "cpu"

    print(f"Python Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"system platform: {sys.platform} {platform.architecture()}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")    
    print(f"Target device is {device}")