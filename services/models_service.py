import os
from typing import Dict, Any, Optional
from utilss.classes.model import Model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

def get_model():
    return None

def save_model():
    return None

def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """Create model based on dataset type."""
    if dataset_str == "imagenet":
        return Model(
            ResNet50(weights="imagenet"), 
            dataset_config["top_k"], 
            dataset_config["min_confidence"], 
            model_path, 
            dataset_config["dataset"]
        )
    elif dataset_str == "cifar100":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model {model_path} does not exist')
    
        resnet_model_cifar100 = tf.keras.models.load_model(model_path)
        print(f"Model {model_path} has been loaded")
        return Model(
            resnet_model_cifar100, 
            dataset_config["top_k"], 
            dataset_config["min_confidence"], 
            model_path, 
            dataset_config["dataset"]
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def delete_model():
    return None

