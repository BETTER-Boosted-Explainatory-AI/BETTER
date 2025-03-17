import os
from fastapi import APIRouter, Depends, HTTPException, Request, status
from request_models.hierarchical_clustering_model import HierarchicalClusterRequest, HierarchicalClusterResult
from typing import Dict
from services.dataset_service import _get_dataset_config, _load_dataset
from services.models_service import _load_model
from services.hierarchical_clusters_service import post_hierarchical_cluster, post_hierarchical_cluster_confusion_matrix

hierarchical_clusters_router = APIRouter()

@hierarchical_clusters_router.post(
    "/hierarchical_clusters", 
    response_model=HierarchicalClusterResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def create_hierarchical_clusters(hierarchical_clusters_data: HierarchicalClusterRequest) -> Dict[str, str]:
    model_filename = hierarchical_clusters_data.model_filename
    graph_type = hierarchical_clusters_data.graph_type
    dataset_str = hierarchical_clusters_data.dataset
    new_hc = post_hierarchical_cluster(model_filename, graph_type, dataset_str)
    if new_hc is None:
        raise HTTPException(status_code=404, detail="Hierarchical Clustering was not created")
    return HierarchicalClusterResult(data=new_hc.tolist())


@hierarchical_clusters_router.post(
    "/hierarchical_clusters/confusion_matrix", 
    response_model=HierarchicalClusterResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def create_hierarchical_clusters(hierarchical_clusters_data: HierarchicalClusterRequest) -> Dict[str, str]:
    model_filename = hierarchical_clusters_data.model_filename
    edges_df_matrix = hierarchical_clusters_data.graph_type
    dataset_str = hierarchical_clusters_data.dataset
    new_hc = post_hierarchical_cluster_confusion_matrix(model_filename, edges_df_matrix, dataset_str)
    if new_hc is None:
        raise HTTPException(status_code=404, detail="Hierarchical Clustering was not created")
    return HierarchicalClusterResult(data=new_hc.tolist())
