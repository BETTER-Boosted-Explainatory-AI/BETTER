import os
from fastapi import APIRouter, HTTPException, status
from request_models.dendrogram_model import DendrogramRequest, DendrogramResult, NamingClusterRequest, NamingClusterResult
from services.dendrogram_service import _get_sub_dendrogram, _rename_cluster

dendrogram_router = APIRouter()

@dendrogram_router.post(
    "/dendrograms", 
    response_model=DendrogramResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_sub_dendrogram(sub_dendrogram_data: DendrogramRequest) -> DendrogramResult:
    user_id = sub_dendrogram_data.user_id
    model_id = sub_dendrogram_data.model_id
    graph_type = sub_dendrogram_data.graph_type
    selected_labels = sub_dendrogram_data.selected_labels
    sub_dendrogram = _get_sub_dendrogram(user_id, model_id, graph_type, selected_labels)
    
    if sub_dendrogram is None:
        raise HTTPException(status_code=404, detail="Sub Hierarchical Clustering was not created")
    
    return DendrogramResult(**sub_dendrogram)

@dendrogram_router.put(
    "/dendrograms/naming_clusters", 
    response_model=NamingClusterResult,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def update_naming(naming_cluster_data: NamingClusterRequest) -> NamingClusterResult:
    cluster_id = naming_cluster_data.cluster_id
    new_name = naming_cluster_data.new_name
    dendrogram_filename = naming_cluster_data.dendrogram_filename
    _rename_cluster(cluster_id, new_name, dendrogram_filename)

    return NamingClusterResult(message="Cluster name updated successfully")
