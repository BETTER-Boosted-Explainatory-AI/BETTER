import os
import shutil
from fastapi import UploadFile

def upload(upload_dir: str, model_file: UploadFile) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    model_path = os.path.join(upload_dir, model_file.filename)

    print(f"Saving model to {model_path}")
    
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)
    
    return model_path