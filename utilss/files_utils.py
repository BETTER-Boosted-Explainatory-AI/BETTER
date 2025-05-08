import os
import shutil
from fastapi import UploadFile
import json
import uuid
import numpy as np
from utilss.classes.user import User
import tensorflow as tf
import importlib.util
from tensorflow.keras.applications.resnet50 import preprocess_input

def upload(upload_dir: str, model_file: UploadFile) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    model_path = os.path.join(upload_dir, model_file.filename)

    print(f"Saving model to {model_path}")
    
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)
    
    return model_path

def upload_model(user_folder: str, model_id: str, model_file: UploadFile, dataset: str, graph_type) -> str:

    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []

    model_id_md = check_models_metadata(models_data, model_id, graph_type)    
    model_subfolder = os.path.join(user_folder,model_id_md)
    os.makedirs(model_subfolder, exist_ok=True)

    if model_id_md is None:
        print(f"Graph type {graph_type} for the model {model_file.filename} already exists. Skipping upload.")
        return os.path.join(model_subfolder, model_file.filename)

    save_model_metadata(models_data, models_json_path, model_id_md, model_file.filename, dataset, graph_type)

    # Save the model file in the model_id subfolder
    model_path = os.path.join(model_subfolder, model_file.filename)
    print(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    return model_path

def save_model_metadata(models_data, models_json_path ,model_id, model_filename, dataset, graph_type) -> None:
        # Prepare model metadata
    model_metadata = {
        "model_id": model_id,
        "file_name": model_filename,
        "dataset": dataset,
        "graph_type": [graph_type],
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
                print(f"Adding '{graph_type}' to graph_type for file '{model_filename}'.")
            else:
                print(f"Graph type '{graph_type}' already exists for file '{model_filename}'. Skipping.")
            break
    else:
        # If no matching model_id is found, append new metadata
        models_data.append(model_metadata)
        print(f"Adding new metadata for file '{model_filename}' with graph type '{graph_type}'.")

    with open(models_json_path, "w") as json_file:
        json.dump(models_data, json_file, indent=4)

def check_models_metadata(models_data, model_id, graph_type):
    for model in models_data:
        if model["model_id"] == model_id and graph_type == model["graph_type"]:
            return None
        elif model["model_id"] == model_id and graph_type != model["graph_type"]:
            return model_id
        else:
            return str(uuid.uuid4())
        
def get_model_info(models_data, model_id):
    for model in models_data:
        print(model)
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
        
def load_numpy_from_directory(directory):
    """
    Load images from a given directory. Assumes images are stored as .npy files.
    """
    print(f"Loading images from {directory}")
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            image = np.load(file_path)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            images.append(image)
    return images

def load_raw_image(file_path):
    """
    Load a raw adversarial example that was saved as a numpy array
    """
    # Load the numpy array and convert back to tensor
    img_example = np.load(file_path)
    return tf.convert_to_tensor(img_example, dtype=tf.float32)

def get_labels_from_dataset_info(dataset_name: str, dataset_path: str) -> list:
    try:
        # Construct the full path to the dataset info file
        dataset_info_file = os.path.join(dataset_path, f"{dataset_name}_info.py")
        
        # Check if the file exists
        if not os.path.exists(dataset_info_file):
            raise FileNotFoundError(f"Dataset info file '{dataset_info_file}' not found.")
        
        # Import the dataset info file as a module
        spec = importlib.util.spec_from_file_location("dataset_info", dataset_info_file)
        dataset_info = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_info)
        
        # Access the labels
        labels = dataset_info.CIFAR100_INFO.get("labels", [])
        return labels
    except Exception as e:
        raise ValueError(f"Error loading labels from dataset info file: {e}")