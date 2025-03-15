from typing import Dict, Any, Optional
from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
from utilss.classes.datasets.dataset_factory import DatasetFactory


def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string."""
    if dataset_str == "cifar100":
        return CIFAR100_INFO
    elif dataset_str == "imagenet":
        return IMAGENET_INFO
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def _load_dataset(dataset_config: Dict[str, Any]):
    """Load the dataset based on configuration."""
    dataset = DatasetFactory.create_dataset(dataset_config["dataset"])
    dataset.load(dataset_config["dataset"])
    return dataset
