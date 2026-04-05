"""
Environment detection and configuration for AirGesture project.

Supports three environments:
- local: Local machine (CPU or GPU)
- hpc: School HPC cluster (SLURM)
- colab: Google Colab
"""

import os


def detect_environment() -> str:
    """
    Detect the current running environment.

    Returns:
        str: "colab", "hpc", or "local"
    """
    # Check for Google Colab
    if os.path.exists("/content/drive"):
        return "colab"

    # Check for HPC cluster (SLURM)
    if "SLURM_JOB_ID" in os.environ:
        return "hpc"

    # Default to local
    return "local"


def get_save_dir(base_name: str = "checkpoints") -> str:
    """
    Get the save directory path based on environment.

    Args:
        base_name: Base name for the save directory

    Returns:
        str: Full path to the save directory
    """
    env = detect_environment()
    if env == "colab":
        return f"/content/drive/MyDrive/{base_name}"
    return base_name


def get_dataset_path() -> str:
    """
    Get the dataset path based on environment.

    For local/HPC: Use DATASET_PATH environment variable or default to data
    For Colab: Use Google Drive path

    Returns:
        str: Full path to the dataset directory
    """
    env = detect_environment()
    if env == "colab":
        return "/content/drive/MyDrive/DSAI5201_Dataset/organized"

    # Local and HPC use environment variable or default
    return os.environ.get("DATASET_PATH", "data")


def setup_environment() -> str:
    """
    Setup the running environment.

    For Colab: Mount Google Drive
    For local/HPC: No special setup needed

    Returns:
        str: The detected environment name
    """
    env = detect_environment()

    if env == "colab":
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except ImportError:
            print("Warning: Running in Colab but google.colab not available")

    return env


def get_device():
    """
    Get the best available device for PyTorch.

    Returns:
        torch.device: CUDA device if available, else CPU
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
