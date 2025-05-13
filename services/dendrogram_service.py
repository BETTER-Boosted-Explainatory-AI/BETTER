import numpy as np
from fastapi import HTTPException, status
from .models_service import _get_model_path
from utilss.classes.dendrogram import Dendrogram
import json

def check_if_dendrogram_exists(user_id, model_id, graph_type):
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="user_id is required"
        )
    
    if model_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="model_id is required"
        )
    
    if graph_type is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="graph_type is required"
        )
    
    model_path = _get_model_path(user_id, model_id)
    if model_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not find model directory"
        )
    
    dendrogram_filename = f'{model_path}/{graph_type}/dendrogram'
    return dendrogram_filename

def _get_sub_dendrogram(user_id, model_id, graph_type, selected_labels):
    dendrogram_filename = check_if_dendrogram_exists(user_id, model_id, graph_type)
    
    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json

def _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name):
    dendrogram_filename = check_if_dendrogram_exists(user_id, model_id, graph_type)

    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    dendrogram.rename_cluster(cluster_id, new_name)
    dendrogram.save_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json
    