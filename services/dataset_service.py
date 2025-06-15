import shutil
import os
from typing import Dict, Any, List, Tuple
import numpy as np
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.s3_connector.s3_dataset_loader import S3DatasetLoader
from utilss.enums.datasets_enum import DatasetsEnum

def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string from S3."""
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.get_dataset_info(dataset_str)


def _load_dataset(dataset_str: str):
    """
    Return a Dataset object populated directly from S3.
    """
    dataset_config = _get_dataset_config(dataset_str)

    dataset_name = dataset_config["dataset"]        # "cifar100" | "imagenet" | …
    dataset = DatasetFactory.create_dataset(dataset_name)
    if dataset_str == DatasetsEnum.CIFAR100.value:
        dataset.load(dataset_name)       

    return dataset


def get_dataset_labels(dataset_str: str) -> List[str]:
    """Get dataset labels from S3."""
    dataset_config = _get_dataset_config(dataset_str)
    return dataset_config["labels"]


def load_single_image(image_key: str) -> bytes:
    """
    Load a single image from S3
    """
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_single_image(image_key)


def load_imagenet_train() -> str:
    """
    Load ImageNet training data with subdirectory structure
    
    Returns:
        str: Path to the local directory with downloaded data
    """
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_imagenet_train()


def load_cifar100_numpy(folder_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-100 data as NumPy arrays
    
    Args:
        folder_type (str): Type of folder (adversarial or clean)
        
    Returns:
        tuple: (images, labels) as NumPy arrays
    """
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_cifar100_numpy(folder_type)


def load_cifar100_meta() -> Dict:
    """
    Load CIFAR-100 meta data
    
    Returns:
        dict: Meta data for CIFAR-100
    """
    bucket_name = os.environ.get('S3_DATASETS_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_DATASETS_BUCKET_NAME environment variable must be set")
        
    s3_loader = S3DatasetLoader(bucket_name=bucket_name)
    
    return s3_loader.load_cifar100_meta()
