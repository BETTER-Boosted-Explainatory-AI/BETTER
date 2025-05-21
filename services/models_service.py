import os
import numpy as np
from typing import Dict, Any, Optional
from utilss.classes.model import Model
from utilss.classes.dendrogram import Dendrogram
import tensorflow as tf
from utilss.photos_utils import preprocess_loaded_image
from services.dataset_service import get_dataset_labels
import boto3
import io
from botocore.exceptions import ClientError
from utilss.enums.datasets_enum import DatasetsEnum
from fastapi import HTTPException, status
import json
import logging
from utilss import debug_utils

# Set up logging
logger = logging.getLogger(__name__)

def get_top_k_predictions(model, image, class_names, top_k=5):
    # Get predictions from the model
    predictions = model.predict(image)

    # Flatten the predictions array if it's 2D (e.g., shape (1, num_classes))
    if len(predictions.shape) == 2:
        predictions = predictions[0]  # Extract the first (and only) batch

    # Get the indices of the top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    # Get the top-k probabilities and corresponding labels
    top_probs = predictions[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    logger.debug(f"Top {top_k} predictions: {list(zip(top_labels, top_probs))}")

    return list(zip(top_labels, top_probs))

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
    
    logger.info(f"Checking model path for user_id: {user_id}, model_id: {model_id}, graph_type: {graph_type}")

    model_path = _get_model_path(user_id, model_id)

    logger.debug(f"Model path: {model_path}")

    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not find model directory"
        )
    return model_path


# def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
#     # Construct the model path based on user_id and model_id
#     BASE_DIR = os.getenv("USERS_PATH", "users")
#     model_path = os.path.join(BASE_DIR, str(user_id), str(model_id))
    
#     # Check if the model path exists
#     if os.path.exists(model_path):
#         return model_path
#     else:
#         logger.debug(f"Model path {model_path} does not exist")
#         return None


def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    """
    Get the model path in S3 based on user_id and model_id
    """
    
    s3_client = boto3.client('s3')
    bucket_name = os.getenv("S3_USERS_BUCKET_NAME")
    
    # Construct the S3 prefix (equivalent to directory path in S3)
    s3_prefix = f"{str(user_id)}/{str(model_id)}/"
    
    # Check if this prefix exists in S3
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_prefix,
            MaxKeys=1 
        )
        
        # If there are objects with this prefix, the "path" exists
        if 'Contents' in response:
            return s3_prefix
        else:
            logger.debug(f"S3 prefix {s3_prefix} does not exist in bucket {bucket_name}")
            return None
    except ClientError as e:
        logger.error(f"Error checking S3 prefix {s3_prefix}: {e}")
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
        

# def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
#     logger.info(f"Loading model {model_path} for dataset {dataset_str}")

#     if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
#         raise ValueError(f"Invalid dataset: {dataset_str}")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f'Model {model_path} does not exist')
    
#     model = tf.keras.models.load_model(model_path)
    
#     print(f"Model {model_path} has been loaded")
#     return Model(
#         model, 
#         dataset_config["top_k"], 
#         dataset_config["min_confidence"], 
#         model_path, 
#         dataset_config["dataset"]
#     )


def _load_model(dataset_str: str, model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """
    Load a model directly from S3 using TensorFlow's built-in S3 support
    """
    import boto3
    from botocore.exceptions import ClientError
    
    logger.info(f"Loading model from S3: {model_path} for dataset {dataset_str}")

    if dataset_str != DatasetsEnum.IMAGENET.value and dataset_str != DatasetsEnum.CIFAR100.value:
        raise ValueError(f"Invalid dataset: {dataset_str}")

    bucket_name = os.getenv("S3_USERS_BUCKET_NAME")
    s3_url = f"s3://{bucket_name}/{model_path}"
    
    try:
        model = tf.keras.models.load_model(s3_url)
        logger.info(f"Model {model_path} has been loaded directly from S3")
        
        return Model(
            model, 
            dataset_config["top_k"], 
            dataset_config["min_confidence"], 
            model_path,
            dataset_config["dataset"]
        )
    except Exception as e:
        logger.error(f"Error loading model directly from S3: {e}")
        raise FileNotFoundError(f'Could not load model from {s3_url}: {e}')




def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
        logger.info(f"Loading model {model_path} for dataset {dataset_config['dataset']}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model {model_path} does not exist')
        resnet_model = tf.keras.models.load_model(model_path)
        return Model(resnet_model, dataset_config["top_k"], dataset_config["min_confidence"], model_path, dataset_config["dataset"])

def delete_model():
    return None

def query_model(top_label, model_id, graph_type, user):
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
        model_path = model_files["model_graph_folder"]
        dendrogram_filename = f'{model_path}/dendrogram'
    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    consistensy = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
    if consistensy is None:
        raise ValueError(f"Label {top_label} not found in dendrogram")
    else:
        logger.debug(f"Top label: {top_label}")
        logger.debug(f"Consistency: {consistensy}")
    return consistensy

def query_predictions(model_id, graph_type, image, user):
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

    dataset = model_info["dataset"]
    labels = get_dataset_labels(dataset)
    model_filename = model_files["model_file"]
    if os.path.exists(model_filename):
        current_model = tf.keras.models.load_model(model_filename)
        logger.debug(f"Model loaded successfully from '{model_filename}'.")
    else:
        raise ValueError(f"Model file {model_filename} does not exist")
    preprocessed_image = preprocess_loaded_image(current_model, image)
    predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
    top_label = predictions[0][0]  # Top label
    top_3_predictions = predictions[:3]  # Top 3 predictions
    return top_label, top_3_predictions


def get_user_models_info(user, model_id):
    models_json_path = user.get_models_json_path()
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
        logger.debug(f"Models data loaded from {models_json_path}")
    else:
        models_data = []
        raise ValueError(f"Models metadata file '{models_json_path}' not found.")
    
    if model_id is None:
        return models_data
    else:
        return get_model_info(models_data, model_id)

def get_model_info(models_data, model_id):
    for model in models_data:
        if str(model["model_id"]) == str(model_id):
            return {
                "model_id": model["model_id"],
                "file_name": model["file_name"],
                "dataset": model["dataset"],
                "graph_type": model["graph_type"],
            }
            
    # If no match is found, return None
    logger.debug(f"Model ID {model_id} not found in models data.")
    return None

def get_model_files(user_folder: str, model_info: dict, graph_type: str):
        logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")

        model_subfolder = os.path.join(user_folder, model_info["model_id"])
        model_file = os.path.join(model_subfolder, model_info["file_name"])
        if not os.path.exists(model_file):
            model_file = None
            raise ValueError(f"Model file {model_file} does not exist")
        
        model_graph_folder = os.path.join(model_subfolder, graph_type)
        if not os.path.exists(model_graph_folder):
            model_graph_folder = None
            raise ValueError(f"Model graph folder {model_graph_folder} does not exist")

        Z_file = os.path.join(model_graph_folder, "dendrogram.pkl")
        if not os.path.exists(Z_file):
            Z_file = None
            logger.debug(f"Z file {Z_file} does not exist")

        dendrogram_file = os.path.join(model_graph_folder, 'dendrogram.json')
        if not os.path.exists(dendrogram_file):
            dendrogram_file = None
            logger.debug(f"Dendrogram file {dendrogram_file} does not exist")

        detector_filename = os.path.join(model_graph_folder, 'logistic_regression_model.pkl')
        if not os.path.exists(detector_filename):
            detector_filename = None
            logger.debug(f"Detector model file {detector_filename} does not exist")

        dataframe_filename = os.path.join(model_graph_folder, 'edges_df.csv')
        if not os.path.exists(dataframe_filename):
            dataframe_filename = None
            logger.debug(f"Dataframe file {dataframe_filename} does not exist")

        return {"model_file": model_file, "Z_file": Z_file, "dendrogram": dendrogram_file, "detector_filename": detector_filename, "dataframe": dataframe_filename, "model_graph_folder": model_graph_folder}
        

