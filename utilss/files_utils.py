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


def upload_model(user_folder: str, model_file: UploadFile, dataset: str, graph_type) -> str:

    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []

    model_id = check_models_metadata(models_data, model_file.filename, graph_type)    

    print(f"model_id: {model_id}")

    if model_id:
        print(f"Graph type {graph_type} for the model {model_file.filename} already exists. Skipping upload.")
        model_subfolder = os.path.join(user_folder,model_id)
        return os.path.join(model_subfolder, model_file.filename)
    else:
        model_id = str(uuid.uuid4())
    
    # Generate a unique model ID
    model_subfolder = os.path.join(user_folder, model_id)
    os.makedirs(model_subfolder, exist_ok=True)

    save_model_metadata(models_data, models_json_path, model_id, model_file.filename, dataset, graph_type)

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
        "graph_type": graph_type,
    }

    # Check if the file_name already exists with a different graph_type
    for model in models_data:
        if model["model_id"] == model_id and model["graph_type"] == graph_type:
            print(f"Metadata for file '{model_filename}' with graph type '{graph_type}' already exists. Skipping.")
            return
        elif model["model_id"] == model_id and model["graph_type"] != graph_type:
            print(f"Updating graph type for file '{model_filename}' to '{graph_type}'.")
            model["graph_type"] = graph_type
            break
    else:
        # If no matching file_name and graph_type found, append new metadata
        models_data.append(model_metadata)
        print(f"Adding new metadata for file '{model_filename}' with graph type '{graph_type}'.")

    with open(models_json_path, "w") as json_file:
        json.dump(models_data, json_file, indent=4)

def check_models_metadata(models_data, model_filename, graph_type):
    for model in models_data:
        if model["file_name"] == model_filename and graph_type != model["graph_type"]:
            return model["model_id"]
    return None