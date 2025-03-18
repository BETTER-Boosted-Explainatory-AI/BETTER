import os
from fastapi import APIRouter, HTTPException, status
from request_models.sub_hierarchical_clusters_model import SubHierarchicalClusterRequest, SubHierarchicalClusterResult
from services.sub_hierarchical_clusters_service import _get_sub_heirarchical_clustering

sub_hierarchical_clusters_router = APIRouter()

@sub_hierarchical_clusters_router.post(
    "/hierarchical_clusters/sub_hierarchical_clusters", 
    response_model=SubHierarchicalClusterResult,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_sub_hierarchical_clustering(sub_hierarchical_clusters_data: SubHierarchicalClusterRequest) -> SubHierarchicalClusterResult:
    dataset = sub_hierarchical_clusters_data.dataset
    selected_labels = sub_hierarchical_clusters_data.selected_labels
    z_filename = sub_hierarchical_clusters_data.z_filename
    sub_hc = _get_sub_heirarchical_clustering(dataset, selected_labels, z_filename)
    if sub_hc is None:
        raise HTTPException(status_code=404, detail="Sub Hierarchical Clustering was not created")
    return SubHierarchicalClusterResult(data=sub_hc)





