from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form

from services.models_service import query_model, _load_model
from services.dataset_service import _get_dataset_config
from request_models.query_model import PredictionResponse
from pathlib import Path
from typing import Dict
import os
import shutil

query_router = APIRouter()

@query_router.post(
    "/query",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    },
    response_model=PredictionResponse  # Use the Pydantic model for response
)
async def upload_images(
    model_filename: str = Form(...), 
    dataset: str = Form(...), 
    image: UploadFile = File(...)  
) -> PredictionResponse:
    try:
        UPLOAD_DIR = os.getenv("UPLOAD_DIR")
        if not UPLOAD_DIR:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Upload directory not configured"
            )

        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Create a unique file path to prevent overwriting
        image_path = Path(UPLOAD_DIR) / f"{os.urandom(16).hex()}_{image.filename}"
        
        # Save the uploaded file
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Process the image
        dataset_config = _get_dataset_config(dataset)
        current_model = _load_model(dataset_config["dataset"], model_filename, dataset_config)
        prediction = query_model(current_model, image_path)
        
        # Return the prediction wrapped in the Pydantic model
        return PredictionResponse(prediction=prediction)

    except Exception as e:
        # Proper error handling
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        ) 
    finally:
        # Ensure file is closed and removed after processing
        if 'image_path' in locals() and os.path.exists(image_path):
            os.unlink(image_path)