from fastapi import HTTPException, status
from .models_service import _check_model_path
from utilss.classes.dendrogram import Dendrogram
import json
from utilss.s3_connector.s3_dataset_utils import get_dataset_config
from .models_service import get_user_models_info

def _get_dendrogram_path(user_id, model_id, graph_type):
    try:
        model_path = _check_model_path(user_id, model_id, graph_type)
        if model_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model path not found"
            )
        dendrogram_filename = f'{model_path}/{graph_type}/dendrogram'
        return dendrogram_filename
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dendrogram path: {str(e)}"
        )


def _get_sub_dendrogram(current_user, model_id, graph_type, selected_labels):
    # Get the S3 path to the dendrogram
    dendrogram_filename = _get_dendrogram_path(current_user.user_id, model_id, graph_type)
    
    if selected_labels is None or selected_labels == []:   
        model_info = get_user_models_info(current_user, model_id)
        if model_info is None or "dataset" not in model_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model info or dataset not found")
        dataset = model_info["dataset"]
        
        if dataset == "cifar100":
            selected_labels = get_dataset_config(dataset)["init_selected_labels"]
        elif dataset == "imagenet":
            selected_labels = get_dataset_config(dataset)["init_selected_labels"]
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Dataset not supported")
    
    try:
        # Create and load dendrogram with S3 support
        dendrogram = Dendrogram(dendrogram_filename)
        dendrogram.load_dendrogram()
        sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
        sub_dendrogram_json = json.loads(sub_dendrogram)
        return sub_dendrogram_json, selected_labels
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name):
    try:
        # Get the S3 path to the dendrogram
        dendrogram_filename = _get_dendrogram_path(user_id, model_id, graph_type)
        # Create and load dendrogram with S3 support
        dendrogram = Dendrogram(dendrogram_filename)
        dendrogram.load_dendrogram()
        result = dendrogram.rename_cluster(cluster_id, new_name)
        if not result:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Cluster rename failed")
        
        dendrogram.save_dendrogram()
        sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
        sub_dendrogram_json = json.loads(sub_dendrogram)

        return sub_dendrogram_json, selected_labels
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def _get_common_ancestor_subtree(current_user, model_id, graph_type, selected_labels):
    if not current_user or not hasattr(current_user, 'user_id'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
        
    if not selected_labels:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No selected labels or invalid number of labels"
        )
    
    print(f"Selected labels: {selected_labels}")
    dendrogram_filename = _get_dendrogram_path(current_user.user_id, model_id, graph_type)
    dendrogram = Dendrogram(dendrogram_filename)
    try:
        dendrogram.load_dendrogram()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    
    try:
        subtree, labels = dendrogram.get_common_ancestor_subtree(selected_labels)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_417_EXPECTATION_FAILED,
            detail=str(e)
        )
    if not subtree or 'id' not in subtree or 'name' not in subtree:
        return None, None
    print(f"Subtree: {subtree}")
    return subtree, labels
