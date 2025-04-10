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


def upload_model(upload_dir: str, model_file: UploadFile, dataset: str, current_user: User) -> str:
    # Ensure the user's folder exists
    user_folder = os.path.join(upload_dir, str(current_user.user_id))
    models_folder = os.path.join(user_folder, "models")
    os.makedirs(models_folder, exist_ok=True)

    # Generate a unique model ID
    model_id = str(uuid.uuid4())
    model_subfolder = os.path.join(models_folder, model_id)
    os.makedirs(model_subfolder, exist_ok=True)

    save_model_metadata(user_folder, model_id, model_file.filename, dataset)

    # Save the model file in the model_id subfolder
    model_path = os.path.join(model_subfolder, model_file.filename)
    print(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    return model_path

def save_model_metadata(user_folder, model_id, model_filename, dataset) -> None:
        # Prepare model metadata
    model_metadata = {
        "model_id": model_id,
        "file_name": model_filename,
        "dataset": dataset,
    }

    # Save metadata to models.json
    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []

    models_data.append(model_metadata)
    print(f"Saving model metadata to {models_json_path}")

    with open(models_json_path, "w") as json_file:
        json.dump(models_data, json_file, indent=4)