import os
import numpy as np
from botocore.exceptions import ClientError
from typing import Dict, Any, Optional
import tensorflow as tf 
import logging
from utilss.classes.model import Model
from utilss.classes.dendrogram import Dendrogram
from utilss.photos_utils import preprocess_loaded_image
from services.dataset_service import get_dataset_labels
from fastapi import HTTPException, status
import json
from utilss.enums.datasets_enum import DatasetsEnum
import tempfile
from utilss.s3_utils import get_users_s3_client

# Set up logging
logger = logging.getLogger(__name__)

# S3 Configuration - Always use S3
S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
if not S3_BUCKET:
    raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

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

    
def delete_model():
    return None

def _get_model_path(user_id: str, model_id: str) -> Optional[str]:
    """
    Get the model path in S3 based on user_id and model_id
    """
    s3_client = get_users_s3_client()
    
    # Construct the S3 prefix (equivalent to directory path in S3)
    s3_prefix = f"{str(user_id)}/{str(model_id)}"
    
    # Check if this prefix exists in S3
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=s3_prefix,
            MaxKeys=1 
        )
        
        # If there are objects with this prefix, the "path" exists
        if 'Contents' in response:
            return s3_prefix
        else:
            logger.debug(f"S3 prefix {s3_prefix} does not exist in bucket {S3_BUCKET}")
            return None
    except ClientError as e:
        logger.error(f"Error checking S3 prefix {s3_prefix}: {e}")
        return None
    
def _get_model_filename(user_id: str, model_id: str, graph_type: str) -> Optional[str]:
    """Get S3 model filename"""
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not find model directory"
        )
    
    # Find .keras file in S3 prefix
    s3_client =  get_users_s3_client() 
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=model_path
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.keras'):
                    return obj['Key']   
        
        return None
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {e}")
        return None
                
def s3_file_exists(bucket_name: str, s3_key: str) -> bool:
    """Check if a file exists in S3"""
    s3_client =  get_users_s3_client() 
    print(f"Checking if file exists in S3: {bucket_name}/{s3_key}")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File found: {bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"File not found: {bucket_name}/{s3_key}, Error: {str(e)}")
        return False

def read_json_from_s3(bucket_name: str, s3_key: str) -> Any:
    """Read a JSON file from S3"""
    s3_client =  get_users_s3_client() 
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        logger.error(f"Error reading JSON from S3 ({bucket_name}/{s3_key}): {e}")
        raise

def load_model_from_s3(bucket_name: str, s3_key: str):
    """Load Keras model from S3"""
    # Check if s3_key is actually a full S3 URI
    if s3_key.startswith('s3://'):
        parts = s3_key.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        bucket = bucket_name
        key = s3_key
    
    s3_client =  get_users_s3_client() 
    
    # Use a temporary directory instead of a temporary file
    # This approach is more reliable on Windows
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, 'model.keras')
        
        try:
            # Download the model file to the temp directory
            s3_client.download_file(bucket, key, temp_model_path)
            
            # Load the model from the temp file
            model = tf.keras.models.load_model(temp_model_path)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from S3 ({bucket}/{key}): {e}")
            raise
        
def construct_model(model_path: str, dataset_config: Dict[str, Any]) -> Model:
    """Load model from S3 and construct Model object"""
    logger.info(f"Loading model {model_path} for dataset {dataset_config['dataset']}")
    
    # Check if model_path is an S3 path or just a key
    if model_path.startswith('s3://'):
        bucket_name = model_path.replace('s3://', '').split('/')[0]
        s3_key = '/'.join(model_path.replace('s3://', '').split('/')[1:])
    else:
        bucket_name = S3_BUCKET
        s3_key = model_path
    
    # Check if model exists in S3
    if not s3_file_exists(bucket_name, s3_key):
        raise FileNotFoundError(f'Model s3://{bucket_name}/{s3_key} does not exist')
    
    # Load model from S3
    resnet_model = load_model_from_s3(bucket_name, s3_key)
    
    return Model(
        resnet_model, 
        dataset_config["top_k"], 
        dataset_config["min_confidence"],
        f"s3://{bucket_name}/{s3_key}", 
        dataset_config["dataset"]
    )
    
def query_model(top_label, model_id, graph_type, user):
    """Query model using dendrogram from S3 without local disk usage"""
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        # model_files = get_model_files(user.get_user_folder(), model_info, graph_type)
        # model_path = model_files["model_graph_folder"]
        model_path = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "graph_folder")
        dendrogram_s3_key = f'{model_path}/dendrogram'
    
    # Use the Dendrogram class with S3 support
    dendrogram = Dendrogram(dendrogram_s3_key)
    dendrogram.load_dendrogram()
    
    consistency = dendrogram.find_name_hierarchy(dendrogram.Z_tree_format, top_label)
    if consistency is None:
        raise ValueError(f"Label {top_label} not found in dendrogram")
    else:
        logger.debug(f"Top label: {top_label}")
        logger.debug(f"Consistency: {consistency}")
    
    return consistency
    
def query_predictions(model_id, graph_type, image, user):
    """Query predictions using model from S3"""
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    # else:
        # model_files = get_model_files(user.get_user_folder(), model_info, graph_type)

    dataset = model_info["dataset"]
    labels = get_dataset_labels(dataset)
    # model_s3_key = model_files["model_file"]
    model_s3_key = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "model_file")
    
    if s3_file_exists(S3_BUCKET, model_s3_key):
        # Load model from S3
        current_model = load_model_from_s3(S3_BUCKET, model_s3_key)
        logger.debug(f"Model loaded successfully from 's3://{S3_BUCKET}/{model_s3_key}'.")
    else:
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_s3_key} does not exist")
    
    preprocessed_image = preprocess_loaded_image(current_model, image)
    predictions = get_top_k_predictions(current_model, preprocessed_image, labels)
    top_label = predictions[0][0]  # Top label
    top_3_predictions = predictions[:3]  # Top 3 predictions
    
    return top_label, top_3_predictions

def get_user_models_info(user, model_id):
    """Get model info from models.json in S3"""
    # Assuming user object has a method to get the models.json path in S3
    # If not, we'll need to construct it
    s3_models_json_key = f"{user.get_user_folder()}/models.json"
    
    if s3_file_exists(S3_BUCKET, s3_models_json_key):
        models_data = read_json_from_s3(S3_BUCKET, s3_models_json_key)
        logger.debug(f"Models data loaded from s3://{S3_BUCKET}/{s3_models_json_key}")
    else:
        models_data = []
        raise ValueError(f"Models metadata file 's3://{S3_BUCKET}/{s3_models_json_key}' not found.")
    
    if model_id is None:
        return models_data
    else:
        return get_model_info(models_data, model_id)


def get_model_info(models_data, model_id):
    for model in models_data:
        if str(model["model_id"]) == str(model_id):
            return model
            
    # If no match is found, return None
    logger.debug(f"Model ID {model_id} not found in models data.")
    return None

def get_model_files(user_folder: str, model_info: dict, graph_type: str):
    """Get model file paths in S3"""
    logger.info(f"Getting model files for user folder: {user_folder}, model info: {model_info}, graph type: {graph_type}")
    
    # Construct S3 paths
    model_subfolder = f"{user_folder}/{model_info['model_id']}"
    model_file = f"{model_subfolder}/{model_info['file_name']}"
    
    if not s3_file_exists(S3_BUCKET, model_file):
        model_file = None
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    
    model_graph_folder = f"{model_subfolder}/{graph_type}"
    
    # Check if the graph folder "exists" by listing objects with this prefix
    s3_client = get_users_s3_client() 
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=model_graph_folder,
        MaxKeys=1
    )
    
    if 'Contents' not in response:
        model_graph_folder = None
        raise ValueError(f"Model graph folder s3://{S3_BUCKET}/{model_graph_folder} does not exist")
    
    # Define file paths
    Z_file = f"{model_graph_folder}/dendrogram.pkl"
    Z_file_exists = s3_file_exists(S3_BUCKET, Z_file)
    if not Z_file_exists:
        Z_file = None
        logger.debug(f"Z file s3://{S3_BUCKET}/{Z_file} does not exist")
    
    dendrogram_file = f"{model_graph_folder}/dendrogram.json"
    dendrogram_file_exists = s3_file_exists(S3_BUCKET, dendrogram_file)
    if not dendrogram_file_exists:
        dendrogram_file = None
        logger.debug(f"Dendrogram file s3://{S3_BUCKET}/{dendrogram_file} does not exist")
    
    detector_filename = f"{model_graph_folder}/logistic_regression_model.pkl"
    detector_file_exists = s3_file_exists(S3_BUCKET, detector_filename)
    if not detector_file_exists:
        detector_filename = None
        logger.debug(f"Detector model file s3://{S3_BUCKET}/{detector_filename} does not exist")
    
    dataframe_filename = f"{model_graph_folder}/edges_df.csv"
    dataframe_file_exists = s3_file_exists(S3_BUCKET, dataframe_filename)
    if not dataframe_file_exists:
        dataframe_filename = None
        logger.debug(f"Dataframe file s3://{S3_BUCKET}/{dataframe_filename} does not exist")
    
    return {
        "model_file": model_file, 
        "Z_file": Z_file, 
        "dendrogram": dendrogram_file, 
        "detector_filename": detector_filename, 
        "dataframe": dataframe_filename, 
        "model_graph_folder": model_graph_folder
    }

def get_model_specific_file(user_folder: str, model_info: dict, graph_type: str, file_type: str):
    model_subfolder = f"{user_folder}/{model_info['model_id']}"
    model_file = f"{model_subfolder}/{model_info['file_name']}"
    
    if not s3_file_exists(S3_BUCKET, model_file):
        raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    
    model_graph_folder = f"{model_subfolder}/{graph_type}"
    s3_client = get_users_s3_client()
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=model_graph_folder,
        MaxKeys=1
    )
    if 'Contents' not in response:
        raise ValueError(f"Model graph folder s3://{S3_BUCKET}/{model_graph_folder} does not exist")

    match file_type:
        case "model_file":
            file_path = model_file
        case "graph_folder":
            file_path = model_graph_folder
        case "Z_file":
            file_path = f"{model_graph_folder}/dendrogram.pkl"
            if not s3_file_exists(S3_BUCKET, file_path):
                logger.debug(f"Z file s3://{S3_BUCKET}/{file_path} does not exist")
                file_path = None
        case "dendrogram":
            file_path = f"{model_graph_folder}/dendrogram.json"
            if not s3_file_exists(S3_BUCKET, file_path):
                logger.debug(f"Dendrogram file s3://{S3_BUCKET}/{file_path} does not exist")
                file_path = None
        case "dataframe":
            file_path = f"{model_graph_folder}/edges_df.csv"
            if not s3_file_exists(S3_BUCKET, file_path):
                logger.debug(f"Dataframe file s3://{S3_BUCKET}/{file_path} does not exist")
                file_path = None
        case _:
            raise ValueError(f"Unknown file_type: {file_type}")

    return file_path
    

    



# def get_detectors_list():
#     model_subfolder = f"{user_folder}/{model_info['model_id']}"
#     model_file = f"{model_subfolder}/{model_info['file_name']}"
    
#     if not s3_file_exists(S3_BUCKET, model_file):
#         model_file = None
#         raise ValueError(f"Model file s3://{S3_BUCKET}/{model_file} does not exist")
    
#     model_graph_folder = f"{model_subfolder}/{graph_type}"
    
#     # Check if the graph folder "exists" by listing objects with this prefix
#     s3_client = get_users_s3_client() 
#     response = s3_client.list_objects_v2(
#         Bucket=S3_BUCKET,
#         Prefix=model_graph_folder,
#         MaxKeys=1
#     )