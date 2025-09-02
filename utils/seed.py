"""
Utility functions for reproducibility and environment setup.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment():
    """Set up environment variables for training."""
    os.environ["WANDB_DISABLE_CODE"] = "true"
    
    # Additional environment setup can be added here
    # e.g., setting CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS, etc.


def get_device(device_index: int = 0) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_index: Index of the CUDA device to use
        
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        device = torch.device(f'cuda:{device_index}')
        print(f"Using CUDA device {device_index}: {torch.cuda.get_device_name(device_index)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


def print_model_stats(model):
    """
    Print model statistics including parameter count.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")