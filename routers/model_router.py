from fastapi import APIRouter, HTTPException, status, Depends
from request_models.model_model import ModelRequest, ModelResult
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from services.models_service import get_user_models_info

model_router = APIRouter()

@model_router.post(
    "/models", 
    response_model=ModelResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_model_info(model_data: ModelRequest, current_user: User = Depends(require_authenticated_user)) -> ModelResult:
    model_id = None
    if model_data:
        model_id = model_data.model_id
        
    model_info = get_user_models_info(current_user, model_id)
    
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelResult(**model_info)
