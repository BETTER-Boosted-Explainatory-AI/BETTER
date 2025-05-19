import os
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from request_models.nma_model import NMAResult
from typing import Dict
from utilss.files_utils import upload_model
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from services.nma_service import (
    _create_nma,
)

nma_router = APIRouter()


@nma_router.post(
    "/nma",
    response_model=NMAResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def create_nma(
    current_user: User = Depends(require_authenticated_user),
    model_file: UploadFile = File(...),
    dataset: str = Form(...),
    graph_type: str = Form(...),
    model_id: str = Form(None),
    min_confidence: float = Form(0.5),
    top_k: int = Form(5),
) -> Dict[str, str]:
    try:
        model_path, model_id_md = upload_model(current_user, model_id, model_file, dataset, graph_type, min_confidence, top_k)
        init_z = _create_nma(model_path, graph_type, dataset, current_user, min_confidence, top_k, model_id_md)
        
        if init_z is None:
            raise HTTPException(status_code=404, detail="Hierarchical Clustering was not created")
        
        return NMAResult(data=init_z.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

