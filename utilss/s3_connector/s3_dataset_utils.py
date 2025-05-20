import os
from typing import Dict, List, Tuple, Any
import numpy as np
from utilss.s3_connector.s3_dataset_loader import S3DatasetLoader

def load_dataset_numpy(dataset_str: str, folder_type: str) -> Dict[str, np.ndarray]:
    """
    Load NumPy data (.npy files) from clean/adversarial folders
    """
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_numpy_data(dataset_str, folder_type)

def load_cifar100_adversarial_or_clean(folder_type: str) -> Dict[str, np.ndarray]:
    return load_dataset_numpy('cifar100', folder_type)

def load_imagenet_adversarial_or_clean(folder_type: str) -> Dict[str, np.ndarray]:
    return load_dataset_numpy('imagenet', folder_type)

def get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string from S3."""
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.get_dataset_info(dataset_str)

def load_dataset_folder(dataset_str: str, folder_type: str) -> str:
    """
    Load a specific folder (clean/adversarial/train) from a dataset
    """
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_folder(dataset_str, folder_type)

def load_single_image(image_key: str) -> bytes:
    """
    Load a single image from S3
    
    Args:
        image_key (str): S3 key for the image
        
    Returns:
        bytes: Image data
    """
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_single_image(image_key)

def load_imagenet_train() -> str:
    """
    Load ImageNet training data with subdirectory structure
    
    Returns:
        str: Path to the local directory with downloaded data
    """
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_imagenet_train()

def load_cifar100_meta() -> Dict:
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_cifar100_meta()

def load_dataset_split(dataset_str: str, split_type: str) -> str:
    """
    Load test or train dataset as is
    """
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    return s3_loader.load_dataset_split(dataset_str, split_type)