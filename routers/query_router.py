from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form

from services.models_service import query_model, _load_model
from services.dataset_service import _get_dataset_config
from utilss.enums.datasets_enum import DatasetsEnum
from request_models.query_model import QueryResponse
from pathlib import Path
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
    response_model=QueryResponse  # Use the Pydantic model for response
)
async def upload_image(
    model_filename: str = Form(...),
    dataset: str = Form(...),
    image: UploadFile = File(...),
    dendrogram_filename: str = Form(...),
) -> QueryResponse:
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
        prediction = current_model.predict(image_path)

        # Extract the top 3 predictions and labels based on dataset type
        if dataset == DatasetsEnum.IMAGENET.value:
            sorted_predictions = sorted(prediction, key=lambda x: x[2], reverse=True)  # Sort by probability
            top_3_predictions = [(p[1], float(p[2])) for p in sorted_predictions[:3]]  # (label, probability)
            top_label = top_3_predictions[0][0]  # Top label
        elif dataset == DatasetsEnum.CIFAR100.value:
            sorted_predictions = sorted(prediction, key=lambda x: x[1], reverse=True)  # Sort by probability
            top_3_predictions = [(dataset_config["labels"][p[0]], float(p[1])) for p in sorted_predictions[:3]]  # (label, probability)
            top_label = top_3_predictions[0][0]  # Top label
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Query result based on the top label
        query_result = query_model(top_label, dendrogram_filename)

        # Return the prediction wrapped in the Pydantic model
        return QueryResponse(query_result=query_result, top_3_predictions=top_3_predictions)

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