from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from services.models_service import query_model, query_predictions
from utilss.files_utils import upload
from request_models.query_model import QueryResponse
import os

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
        PHOTOS_DIR = os.getenv("UPLOAD_DIR")
        if not PHOTOS_DIR:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Upload photos directory not configured"
            )

        image_path = upload(PHOTOS_DIR, image)
        top_label, top_3_predictions = query_predictions(dataset, model_filename, image_path)
        query_result = query_model(top_label, dendrogram_filename)

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