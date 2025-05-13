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


def get_model():
    return None

def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    # Construct the model path based on user_id and model_id
    BASE_DIR = os.getenv("USERS_PATH", "users")
    model_path = os.path.join(BASE_DIR, str(user_id), str(model_id))
    
    # Check if the model path exists
    if os.path.exists(model_path):
        return model_path
    else:
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


def get_preprocess_function(model):
    print("Determining preprocessing function based on model configuration...")
    preprocess_map = {
        "resnet50": resnet50_preprocess,
        "vgg16": vgg16_preprocess,
        "inception_v3": inception_v3_preprocess,
        "mobilenet": mobilenet_preprocess,
        "efficientnet": efficientnet_preprocess,
        "xception": xception_preprocess,
    }

    # Check the model's configuration for a match
    model_config = model.get_config()
    if "name" in model_config:
        model_name = model_config["name"].lower()
        print(f"Model name: {model_name}")
        for key in preprocess_map.keys():
            if key in model_name:
                print(f"Detected model type: {key}")
                return preprocess_map[key]

    for layer in model.layers:
        layer_name = layer.name.lower()
        print(f"Checking layer: {layer_name}")
        for model_name in preprocess_map.keys():
            if model_name in layer_name:
                print(f"Detected model type: {model_name}")
                return preprocess_map[model_name]

    # If no matching model type is found, use generic normalization
    print("No supported model type found in the configuration. Falling back to generic normalization.")
    return lambda x: x / 255.0  # Generic normalization to [0, 1]


# Cached preprocessing function
_cached_preprocess_function = {}

def get_cached_preprocess_function(model):
    """
    Get the cached preprocessing function for the given model.
    If not cached, fetch it and store it in the cache.
    """
    global _cached_preprocess_function
    model_id = id(model)  # Use the model's unique ID as the cache key
    if model_id not in _cached_preprocess_function:
        _cached_preprocess_function[model_id] = get_preprocess_function(model)
    return _cached_preprocess_function[model_id]

