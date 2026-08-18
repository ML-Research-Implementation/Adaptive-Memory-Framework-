"""
Utility functions for AMMR framework.
Includes device management, reproducibility, logging, and helper functions.
"""

import random
import numpy as np
import torch
from typing import Optional
from config import DEVICE, SEED, HEADER_WIDTH, ENABLE_DETAILED_LOGGING


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_reproducibility() -> None:
    """Initialize reproducibility settings using config seed."""
    set_seed(SEED)


# =====================================================================
# LOGGING AND FORMATTING
# =====================================================================

def print_header(title: str) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Title text for the header.
    """
    if not ENABLE_DETAILED_LOGGING:
        return
    
    print()
    print("=" * HEADER_WIDTH)
    print(title)
    print("=" * HEADER_WIDTH)


def print_section(title: str, content: str) -> None:
    """
    Print a formatted section with title and content.
    
    Args:
        title: Section title.
        content: Section content to display.
    """
    if not ENABLE_DETAILED_LOGGING:
        return
    
    print_header(title)
    print(content)


# =====================================================================
# MODEL UTILITIES
# =====================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Total number of parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters (requires_grad=True) in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def freeze_model(model: torch.nn.Module) -> None:
    """
    Freeze all parameters in a model (set requires_grad=False).
    
    Args:
        model: PyTorch model to freeze.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_model(model: torch.nn.Module) -> None:
    """
    Unfreeze all parameters in a model (set requires_grad=True).
    
    Args:
        model: PyTorch model to unfreeze.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True


def get_device() -> torch.device:
    """
    Get the default device for computations.
    
    Returns:
        torch.device: CUDA if available, else CPU.
    """
    return DEVICE


# =====================================================================
# TENSOR UTILITIES
# =====================================================================

def to_device(tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Move tensor to specified device.
    
    Args:
        tensor: PyTorch tensor.
        device: Target device. If None, uses default device.
        
    Returns:
        Tensor on target device.
    """
    if device is None:
        device = DEVICE
    return tensor.to(device)


def create_mask(
    sequence_length: int,
    valid_indices: list,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a boolean mask from valid indices.
    
    Args:
        sequence_length: Total sequence length.
        valid_indices: List of valid indices to mark as True.
        device: Target device.
        
    Returns:
        Boolean tensor of shape (sequence_length,).
    """
    if device is None:
        device = DEVICE
    
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    for idx in valid_indices:
        mask[idx] = True
    return mask


# =====================================================================
# CHECKPOINT UTILITIES
# =====================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        step: Current training step.
        checkpoint_path: Path to save checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into.
        optimizer: Optimizer to load state into (optional).
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Saved training step.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from {checkpoint_path} at step {step}")
    return step


# =====================================================================
# STATISTICS UTILITIES
# =====================================================================

def calculate_statistics(values: list) -> dict:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values.
        
    Returns:
        Dictionary with min, max, mean, std.
    """
    arr = np.array(values)
    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
    }


def format_number(value: float, decimals: int = 4) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places.
        
    Returns:
        Formatted string.
    """
    return f"{value:.{decimals}f}"
