from typing import Dict, Any, Optional, List
from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.enums.datasets_enum import DatasetsEnum
import shutil
from utilss.s3_datasets_connector.s3_dataset_loader import S3DatasetLoader


def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        return CIFAR100_INFO
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        return IMAGENET_INFO
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def _load_dataset(dataset_config: Dict[str, Any]):
    """Load the dataset based on configuration."""
    dataset = DatasetFactory.create_dataset(dataset_config["dataset"])
    dataset.load(dataset_config["dataset"])
    
    return dataset

def _get_dataset_labels(dataset_str: str) -> List[str]:
    """Get dataset labels based on dataset string."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        return CIFAR100_INFO["labels"]
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        return IMAGENET_INFO["labels"]
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")
    
# def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
#     """Get dataset configuration based on dataset string."""
#     if dataset_str == DatasetsEnum.CIFAR100.value:
#         return CIFAR100_INFO.copy()
#     elif dataset_str == DatasetsEnum.IMAGENET.value:
#         return IMAGENET_INFO.copy()
#     else:
#         raise ValueError(f"Invalid dataset: {dataset_str}")

# def _load_dataset(dataset_config: Dict[str, Any]):
#     """Load the dataset from S3."""
#     dataset_name = dataset_config["dataset"]
    
#     # Create the dataset
#     dataset = DatasetFactory.create_dataset(dataset_name)
    
#     # Always use S3 for loading
#     s3_loader = S3DatasetLoader()
#     temp_dir = s3_loader.load_from_s3(dataset_name)
    
#     try:
#         # Load the dataset from the temp directory
#         dataset.load(dataset_name, data_dir=temp_dir)
#     finally:
#         # Clean up the temp directory
#         shutil.rmtree(temp_dir)
    
#     return dataset

# def _get_dataset_labels(dataset_str: str) -> List[str]:
#     """Get dataset labels based on dataset string."""
#     if dataset_str == DatasetsEnum.CIFAR100.value:
#         return CIFAR100_INFO["labels"]
#     elif dataset_str == DatasetsEnum.IMAGENET.value:
#         return IMAGENET_INFO["labels"]
#     else:
#         raise ValueError(f"Invalid dataset: {dataset_str}")