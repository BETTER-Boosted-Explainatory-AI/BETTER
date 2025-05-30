from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from services.models_service import query_model, query_predictions
from request_models.query_model import QueryResponse
from utilss.classes.user import User
from services.users_service import require_authenticated_user
from utilss.photos_utils import encode_image_to_base64
from PIL import Image, UnidentifiedImageError
import io
import numpy as np

query_router = APIRouter()

@query_router.post(
    "/api/query",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    },
    response_model=QueryResponse  # Use the Pydantic model for response
)
async def verbal_explaination_query(
    current_model_id: str = Form(...),
    graph_type: str = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(require_authenticated_user)
):
    try:
        if image.size is 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image file is required."
            )

        if not graph_type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Graph type is required."
            )
        
        image_content = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded file is not a valid image."
            )
        
        top_label, top_predictions = query_predictions(current_model_id, graph_type, image_content, current_user)
        query_result = query_model(top_label, current_model_id, graph_type, current_user)
        image_array = np.array(pil_image)
        image_base64 = encode_image_to_base64(image_array)

        return QueryResponse(query_result=query_result, top_predictions=top_predictions, image=image_base64)
    
    except HTTPException as http_exc:
    # Re-raise HTTPException to preserve its status code and message
        raise http_exc
    
    except Exception as e:
        # Proper error handling
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        ) 
