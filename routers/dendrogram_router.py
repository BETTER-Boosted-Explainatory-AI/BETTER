from fastapi import APIRouter, HTTPException, status, Depends
from request_models.dendrogram_model import DendrogramRequest, DendrogramResult, NamingClusterRequest
from services.dendrogram_service import _get_sub_dendrogram, _rename_cluster
from services.users_service import require_authenticated_user
from utilss.classes.user import User

dendrogram_router = APIRouter()

@dendrogram_router.post(
    "/api/dendrograms", 
    response_model=DendrogramResult,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_sub_dendrogram(sub_dendrogram_data: DendrogramRequest, current_user: User = Depends(require_authenticated_user)) -> DendrogramResult:
    model_id = sub_dendrogram_data.model_id
    graph_type = sub_dendrogram_data.graph_type
    selected_labels = sub_dendrogram_data.selected_labels
    sub_dendrogram, selected_labels = _get_sub_dendrogram(current_user, model_id, graph_type, selected_labels)
    
    if sub_dendrogram is None:
        raise HTTPException(status_code=404, detail="Sub Hierarchical Clustering was not created")
    
    return DendrogramResult(**sub_dendrogram, selected_labels=selected_labels)

@dendrogram_router.put(
    "/api/dendrograms/naming_clusters", 
    response_model=DendrogramResult,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def update_naming(naming_cluster_data: NamingClusterRequest, current_user: User = Depends(require_authenticated_user)) -> DendrogramResult:
    model_id = naming_cluster_data.model_id
    graph_type = naming_cluster_data.graph_type
    selected_labels = naming_cluster_data.selected_labels
    cluster_id = naming_cluster_data.cluster_id
    new_name = naming_cluster_data.new_name
    user_id = current_user.user_id

    sub_dendrogram = _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name)
    if sub_dendrogram is None:
        raise HTTPException(status_code=404, detail="Sub Hierarchical Clustering was not created")
    return DendrogramResult(**sub_dendrogram)
