import os
from fastapi import APIRouter, HTTPException, status
from request_models.hierarchical_clustering_model import HierarchicalClusterResult
from request_models.confusion_matrix_model import ConfusionMatrixRequest
from typing import Dict

from services.confusion_matrix_service import post_hierarchical_cluster_confusion_matrix


confusion_matrix_router = APIRouter()

@confusion_matrix_router.post(
    "/hierarchical_clusters/confusion_matrix", 
    response_model=HierarchicalClusterResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def create_confusion_matrix(confusion_metrix_data: ConfusionMatrixRequest) -> Dict[str, str]:
    model_filename = confusion_metrix_data.model_filename
    edges_df_filename = confusion_metrix_data.edges_df_filename
    dataset_str = confusion_metrix_data.dataset
    new_hc = post_hierarchical_cluster_confusion_matrix(model_filename, edges_df_filename, dataset_str)
    if new_hc is None:
        raise HTTPException(status_code=404, detail="Hierarchical Clustering was not created")
    return HierarchicalClusterResult(data=new_hc.tolist())
