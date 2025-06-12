import os
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from request_models.nma_model import NMAResult
from typing import Dict
from utilss.files_utils import upload_model, _update_model_metadata, user_has_job_running
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from utilss.enums.graph_types import GraphTypes
from utilss.enums.graph_types import GraphTypes
from utilss.aws_job_utils import submit_nma_batch_job

nma_router = APIRouter()

def _validate_graph_type(graph_type: str):
    if graph_type not in [
        GraphTypes.SIMILARITY.value,
        GraphTypes.DISSIMILARITY.value,
        GraphTypes.COUNT.value,
    ]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Graph type must be either 'similarity', 'dissimilarity', or 'count'"
        )
        
def _handle_nma_submission(
    current_user: User,
    dataset: str,
    graph_type: str,
    min_confidence: float,
    top_k: int,
    model_id: str = None,
    model_file: UploadFile = None
) -> NMAResult:
    _validate_graph_type(graph_type)
    if model_file is None and model_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model file or model_id is required"
        )
        
        
    has_running_job = user_has_job_running(current_user)
    if has_running_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already has a running NMA job. Please wait for it to finish before submitting a new one."
        )
    
    model_filename = model_file.filename if model_file else None
    if model_file is not None:
        model_path, model_id_md = upload_model(
            current_user, model_id, model_file, graph_type)
        model_id_f = model_id_md
    else:
        model_id_f = model_id 
    job_id = submit_nma_batch_job(current_user.user_id, model_id_f, dataset, graph_type, min_confidence, top_k)        
    
    print(f"Submitting NMA job with parameters: {current_user.user_id}, {model_filename},{graph_type}")
    
    print(f"Submitted NMA job with ID: {job_id}")
    metadata_result = _update_model_metadata(
        current_user, model_id_f, model_filename, dataset, graph_type, min_confidence, top_k, job_id)
    if not metadata_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model metadata"
        )
    message = {"message": "NMA job has been submitted successfully."}
    return NMAResult(**message)



@nma_router.post(
    "/api/nma",
    response_model=NMAResult,
    status_code=status.HTTP_202_ACCEPTED,
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
    min_confidence: float = Form(0.5),
    top_k: int = Form(5),
) -> NMAResult:
    try:
        return _handle_nma_submission(
            current_user, dataset, graph_type, min_confidence, top_k, model_file=model_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@nma_router.post(
    "/api/nma/{model_id}",
    response_model=NMAResult,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def create_nma_by_id(
    model_id: str,
    current_user: User = Depends(require_authenticated_user),
    dataset: str = Form(...),
    graph_type: str = Form(...),
    min_confidence: float = Form(0.5),
    top_k: int = Form(5),
) -> NMAResult:
    try:
        return _handle_nma_submission(
            current_user, dataset, graph_type, min_confidence, top_k, model_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
