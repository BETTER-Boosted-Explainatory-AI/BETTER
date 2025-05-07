import os
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from request_models.nma_model import NMARequest, NMAResult
from typing import Dict
from services.TBD_hierarchical_clusters_service import post_hierarchical_cluster, post_new_hierarchical_cluster
from utilss.files_utils import upload, upload_model
from services.users_service import get_current_session_user
from utilss.classes.user import User
import shutil

nma_router = APIRouter()



@nma_router.post(
    "/hierarchical_clusters", 
    response_model=NMAResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def create_nma(nma_data: NMARequest) -> Dict[str, str]:
    user_id = nma_data.user_id
    model_file = nma_data.model_file
    dataset_name = nma_data.dataset
    graph_type = nma_data.graph_type
    min_confidence = nma_data.min_confidence
    top_k = nma_data.top_k
