import os
from typing import Dict, Any, Optional
from utilss.classes.model import Model
from utilss.classes.dendrogram import Dendrogram
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from utilss.enums.datasets_enum import DatasetsEnum
from services.dataset_service import _get_dataset_config
from fastapi import HTTPException, status
import json

def get_model():
    return None

def _check_model_path(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="user_id is required"
        )
    
    if model_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="model_id is required"
        )
    
    if graph_type is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="graph_type is required"
        )
    
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not find model directory"
        )
    return model_path


def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    # Construct the model path based on user_id and model_id
    BASE_DIR = os.getenv("USERS_PATH", "users")
    model_path = os.path.join(BASE_DIR, str(user_id), str(model_id))
    
    # Check if the model path exists
    if os.path.exists(model_path):
        return model_path
    else:
        return None

def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not find model directory"
        )
    for file_name in os.listdir(model_path):
        if file_name.endswith(".keras"):
            model_filename = os.path.join(model_path, file_name)
            return model_filename
        

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

def get_preprocess_function(model):
    print("Determining preprocessing function based on model layers...")
    preprocess_map = {
        "resnet50": resnet50_preprocess,
        "vgg16": vgg16_preprocess,
        "inception_v3": inception_v3_preprocess,
        "mobilenet": mobilenet_preprocess,
        "efficientnet": efficientnet_preprocess,
        "xception": xception_preprocess,
    }

    for layer in model.layers:
        layer_name = layer.name.lower()
        for model_name in preprocess_map.keys():
            if model_name in layer_name:
                print(f"Detected model type: {model_name}")
                return preprocess_map[model_name]

    # If no matching model type is found, use generic normalization
    print("No supported model type found in the layers. Falling back to generic normalization.")
    return lambda x: x / 255.0  # Generic normalization to [0, 1]

