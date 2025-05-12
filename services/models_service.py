import os
from typing import Dict, Any, Optional
from utilss.classes.model import Model
from utilss.classes.dendrogram import Dendrogram
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from utilss.enums.datasets_enum import DatasetsEnum
from services.dataset_service import _get_dataset_config

def get_model():
    return None

def save_model():
    return None

def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    print(f"Loading model {model_path} for dataset {dataset_str}")

    if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
        raise ValueError(f"Invalid dataset: {dataset_str}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model {model_path} does not exist')

    model = tf.keras.models.load_model(model_path)
    print(f"Model {model_path} has been loaded")
    return Model(
        model, 
        dataset_config["top_k"], 
        dataset_config["min_confidence"], 
        model_path, 
        dataset_config["dataset"]
    )


def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
        print(f"Loading model {model_path} for dataset {dataset_config['dataset']}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model {model_path} does not exist')
        resnet_model = tf.keras.models.load_model(model_path)
        return Model(resnet_model, dataset_config["top_k"], dataset_config["min_confidence"], model_path, dataset_config["dataset"])

def delete_model():
    return None

def query_model(top_label: str, dendrogram_filenaem: str):
    dendrogram = Dendrogram(dendrogram_filenaem)
    dendrogram.load_dendrogram_from_json()
    consistensy = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
    print(f"Consistency for {top_label}: {consistensy}")
    return consistensy

def query_predictions(dataset, model_filename, image_path):
    dataset_config = _get_dataset_config(dataset)
    current_model = _load_model(dataset_config["dataset"], model_filename, dataset_config)
    prediction = current_model.predict(image_path)

    # Extract the top 3 predictions and labels based on dataset type
    if dataset == DatasetsEnum.IMAGENET.value:
        sorted_predictions = sorted(prediction, key=lambda x: x[2], reverse=True)  # Sort by probability
        top_3_predictions = [(p[1], float(p[2])) for p in sorted_predictions[:3]]  # (label, probability)
        top_label = top_3_predictions[0][0]  # Top label
        return top_label, top_3_predictions
    elif dataset == DatasetsEnum.CIFAR100.value:
        sorted_predictions = sorted(prediction, key=lambda x: x[1], reverse=True)  # Sort by probability
        top_3_predictions = [(dataset_config["labels"][p[0]], float(p[1])) for p in sorted_predictions[:3]]  # (label, probability)
        top_label = top_3_predictions[0][0]  # Top label
        return top_label, top_3_predictions
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    

