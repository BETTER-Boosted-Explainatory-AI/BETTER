import os
from typing import Dict, Any, Optional
from utilss.classes.model import Model
from utilss.classes.dendrogram import Dendrogram
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from utilss.enums.datasets_enum import DatasetsEnum

def get_model():
    return None

def save_model():
    return None

def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """Create model based on dataset type."""
    MODELS_PATH = os.getenv("MODELS_PATH")
    model_file_path = f'{MODELS_PATH}{model_path}.keras'
    print(f"Loading model {model_file_path} for dataset {dataset_str}")

    if dataset_str == DatasetsEnum.IMAGENET.value:
        return Model(
            ResNet50(weights="imagenet"), 
            dataset_config["top_k"], 
            dataset_config["min_confidence"], 
            model_path, 
            dataset_config["dataset"]
        )
    elif dataset_str == DatasetsEnum.CIFAR100.value:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f'Model {model_path} does not exist')
    
        resnet_model_cifar100 = tf.keras.models.load_model(model_file_path)
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

def query_model(top_label: str, dendrogram_filenaem: str):
    dendrogram = Dendrogram(dendrogram_filenaem)
    dendrogram.load_dendrogram_from_json()
    consistensy = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
    print(f"Consistency for {top_label}: {consistensy}")
    return consistensy

