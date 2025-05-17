from fastapi import APIRouter, HTTPException, status, Depends
from request_models.model_model import ModelRequest, ModelResult
from services.users_service import get_current_session_user
from utilss.classes.user import User
from services.models_service import get_user_models_info
from typing import List

model_router = APIRouter()

@model_router.get(
    "/models", 
    response_model=List[ModelResult], 
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_model_info(current_user: User = Depends(get_current_session_user)) -> List[ModelResult]:
    models_info = get_user_models_info(current_user, None)
    
    if models_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return models_info


@model_router.get(
    "/models/current", 
    response_model=ModelRequest, 
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_current_model_info(current_user: User = Depends(get_current_session_user)) -> ModelRequest:
    curr_model_info = current_user.get_current_model()
    
    if curr_model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return curr_model_info

@model_router.put(
    "/models/current", 
    response_model=ModelRequest, 
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)  
async def set_current_model_info(model: ModelRequest, current_user: User = Depends(get_current_session_user)) -> ModelRequest:
    model_dict = model.model_dump(exclude_unset=True) 
    curr_model_info = current_user.set_current_model(model_dict)
    
    if curr_model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return curr_model_info
