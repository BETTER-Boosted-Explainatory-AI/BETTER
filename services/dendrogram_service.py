import numpy as np
from fastapi import HTTPException, status
from .models_service import _check_model_path
from utilss.classes.dendrogram import Dendrogram
import json

def _get_dendrogram_path(user_id, model_id, graph_type):
    model_path = _check_model_path(user_id, model_id, graph_type)
    dendrogram_filename = f'{model_path}/{graph_type}/dendrogram'
    return dendrogram_filename

def _get_sub_dendrogram(user_id, model_id, graph_type, selected_labels):
    dendrogram_filename = _get_dendrogram_path(user_id, model_id, graph_type)
    
    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json

def _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name):
    dendrogram_filename = _get_dendrogram_path(user_id, model_id, graph_type)

    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    dendrogram.rename_cluster(cluster_id, new_name)
    dendrogram.save_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json
    