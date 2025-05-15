import os
import shutil
from fastapi import UploadFile
import json
import uuid
import numpy as np
import tensorflow as tf
from utilss.photos_utils import preprocess_numpy_image

# def upload(upload_dir: str, model_file: UploadFile) -> str:
#     os.makedirs(upload_dir, exist_ok=True)
#     model_path = os.path.join(upload_dir, model_file.filename)

#     print(f"Saving model to {model_path}")

#     with open(model_path, "wb") as f:
#         shutil.copyfileobj(model_file.file, f)

#     return model_path


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
