import os
import shutil
from fastapi import UploadFile
import json
import uuid
import numpy as np
import tensorflow as tf
from services.models_service import get_cached_preprocess_function
import importlib.util
from utilss.uuid_utils import is_valid_uuid
from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
from PIL import Image
import io

def upload(upload_dir: str, model_file: UploadFile) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    model_path = os.path.join(upload_dir, model_file.filename)

    print(f"Saving model to {model_path}")

    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    return model_path


def upload_model(
    user_folder: str,
    model_id: str,
    model_file: UploadFile,
    dataset: str,
    graph_type: str,
    min_confidence: float,
    top_k: int,
) -> str:
   
    filename = os.path.basename(model_file.filename)
    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []

    try:
        model_id_md = check_models_metadata(models_data, model_id, graph_type)
    except ValueError as e:
        print(str(e))
        raise 

    model_subfolder = os.path.join(user_folder, model_id_md)
    os.makedirs(model_subfolder, exist_ok=True)
    
    save_model_metadata(
        models_data,
        models_json_path,
        model_id_md,
        model_file.filename,
        dataset,
        graph_type,
        min_confidence,
        top_k,
    )
    
    model_path = f'{model_subfolder}/{filename}'
    print(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    return model_path



def save_model_metadata(
    models_data, models_json_path, model_id, model_filename, dataset, graph_type, min_confidence, top_k,
) -> None:
    # Prepare model metadata
    model_metadata = {
        "model_id": model_id,
        "file_name": model_filename,
        "dataset": dataset,
        "graph_type": [graph_type],
        "min_confidence": min_confidence,
        "top_k": top_k
    }

    # Check if the model_id already exists
    for model in models_data:
        if model["model_id"] == model_id:
            # If graph_type is not already a list, convert it to a list
            if not isinstance(model["graph_type"], list):
                model["graph_type"] = [model["graph_type"]]

            # Add the new graph_type if it doesn't already exist
            if graph_type not in model["graph_type"]:
                model["graph_type"].append(graph_type)
                print(
                    f"Adding '{graph_type}' to graph_type for file '{model_filename}'."
                )
            else:
                print(
                    f"Graph type '{graph_type}' already exists for file '{model_filename}'. Skipping."
                )
            break
    else:
        # If no matching model_id is found, append new metadata
        models_data.append(model_metadata)
        print(
            f"Adding new metadata for file '{model_filename}' with graph type '{graph_type}'."
        )

    with open(models_json_path, "w") as json_file:
        json.dump(models_data, json_file, indent=4)


def check_models_metadata(models_data, model_id, graph_type):
    for model in models_data:  
        if model.get("model_id") == model_id:
            if graph_type in model.get("graph_type", []):
                raise ValueError(
                    f"Graph type '{graph_type}' for model ID '{model_id}' already exists."
                )
            return model_id
    return str(uuid.uuid4())
        
def get_user_models_info(user_folder: str, model_id: str):
    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []
        raise ValueError(f"Models metadata file '{models_json_path}' not found.")
    
    if model_id is None:
        return models_data
    else:
        return get_model_info(models_data, model_id)

def get_model_info(models_data, model_id):
    for model in models_data:
        if model["model_id"] == model_id:
            return {
                "model_id": model["model_id"],
                "file_name": model["file_name"],
                "dataset": model["dataset"],
                "graph_type": model["graph_type"],
            }
    # If no match is found, return None
    print(f"Model {model_id} doesn't exist.")
    return None

def get_model_files(user_folder: str, model_info: dict, graph_type: str):
        model_subfolder = os.path.join(user_folder, model_info["model_id"])
        model_file = os.path.join(model_subfolder, model_info["file_name"])
        if not os.path.exists(model_file):
            model_file = None
            raise ValueError(f"Model file {model_file} does not exist")
        model_graph_folder = os.path.join(model_subfolder, graph_type)
        Z_file = os.path.join(model_graph_folder, "dendrogram.pkl")
        if not os.path.exists(Z_file):
            Z_file = None
            print(f"Z file {Z_file} does not exist")
        detector_filename = os.path.join(model_graph_folder, 'logistic_regression_model.pkl')
        if not os.path.exists(detector_filename):
            detector_filename = None
            print(f"Detector model file {detector_filename} does not exist")
        dataframe_filename = os.path.join(model_graph_folder, 'edges_df.csv')
        if not os.path.exists(dataframe_filename):
            dataframe_filename = None
            print(f"Dataframe file {dataframe_filename} does not exist")
        return {"model_file": model_file, "Z_file": Z_file, "detector_filename": detector_filename, "dataframe": dataframe_filename, "model_graph_folder": model_graph_folder}
        
def load_numpy_from_directory(model ,directory):
    """
    Load images from a given directory. Assumes images are stored as .npy files.
    """
    print(f"Loading images from {directory}")
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            image = np.load(file_path)
            preprocess_image = preprocess_numpy_image(model, image)
            images.append(preprocess_image)
    return images

def load_raw_image(file_path):
    """
    Load a raw adversarial example that was saved as a numpy array
    """
    # Load the numpy array and convert back to tensor
    img_example = np.load(file_path)
    return tf.convert_to_tensor(img_example, dtype=tf.float32)

def get_labels_from_dataset_info(dataset_name: str) -> list:
    try:
        if dataset_name == "cifar100":
            return CIFAR100_INFO.get("labels", [])
        elif dataset_name == "imagenet":
            return IMAGENET_INFO.get("labels", [])
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    except Exception as e:
        raise ValueError(f"Error loading labels from dataset info file: {e}")

def preprocess_image(model, image):
    expected_shape = model.input_shape
    input_height, input_width = expected_shape[1], expected_shape[2]
    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    pil_image = pil_image.resize((input_width, input_height))
    preprocess_input = get_cached_preprocess_function(model)
    image_array = preprocess_input(np.array(pil_image))
    image_preprocessed = np.expand_dims(image_array, axis=0)
    return image_preprocessed

def preprocess_numpy_image(model, image):
    """
    Preprocess a NumPy array image for the given model.
    """
    if image.ndim == 3:
        # If the image is 3D, add a batch dimension
        image = np.expand_dims(image, axis=0)

    # Get the appropriate preprocessing function for the model
    preprocess_input = get_cached_preprocess_function(model)

    # Apply the preprocessing function
    image_preprocessed = preprocess_input(image)

    return image_preprocessed
