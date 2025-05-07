import os
import shutil
from fastapi import UploadFile
import json
import uuid
from utilss.classes.user import User

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