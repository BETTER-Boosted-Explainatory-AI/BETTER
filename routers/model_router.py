from fastapi import APIRouter, HTTPException, status, Depends, Query
from services.users_service import require_authenticated_user
from request_models.model_model import ModelRequest, ModelsResult, CurrentModelRequest
from utilss.classes.user import User
from services.models_service import get_user_models_info
from typing import List

model_router = APIRouter()

@model_router.get(
    "/models",
    response_model=List[ModelsResult],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_model_info(
    current_user: User = Depends(require_authenticated_user),
    status_filter: str = Query(None, alias="status")
) -> List[ModelsResult]:
    models_info = get_user_models_info(current_user, None)
    if models_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if status_filter == "succeeded":
        filtered_models = []
        for model in models_info:
            succeeded_types = [
                job["job_graph_type"]
                for job in model.get("batch_jobs", [])
                if job.get("job_status") == "succeeded"
            ]
            if succeeded_types:
                model_copy = model.copy()
                model_copy["graph_type"] = list(set(succeeded_types))
                filtered_models.append(model_copy)
        return filtered_models
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
async def get_current_model_info(current_user: User = Depends(require_authenticated_user)) -> ModelRequest:
    curr_model_info = current_user.get_current_model()

    if curr_model_info is None or not curr_model_info.get("model_id"):
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
async def set_current_model_info(model: CurrentModelRequest, current_user: User = Depends(require_authenticated_user)) -> ModelRequest:
    model_dict = model.model_dump(exclude_unset=True) 
    models_info = get_user_models_info(current_user, model.model_id)
    model_dict = {
        "model_id": models_info["model_id"],
        "file_name": models_info["file_name"],
        "dataset": models_info["dataset"],
        "graph_type": model_dict.get("graph_type")
    }
    
    curr_model_info = current_user.set_current_model(model_dict)
    
    if curr_model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return curr_model_info
