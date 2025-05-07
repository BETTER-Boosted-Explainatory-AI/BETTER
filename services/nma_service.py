from fastapi import HTTPException, status
from .dataset_service import _get_dataset_config, _load_dataset
from .graphs_service import _create_graph
from .models_service import _load_model, construct_model
import numpy as np
from utilss.classes.nma import NMA
from utilss.classes.edges_dataframe import EdgesDataframe
from utilss.classes.dendrogram import Dendrogram

def _create_nma(model_filename, graph_type, dataset_str, user_id, min_confidence, top_k):
    if model_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
        
    if graph_type != "similarity" and graph_type != "dissimilarity":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity' or 'dissimilarity'"
        )
    
    dataframe_filename = f'edges_{graph_type}_{model_filename}.csv'
    dendrogram_filename = f'dendrogram_{graph_type}_{model_filename}.json'
    
    # Get dataset configuration
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        loaded_model = _load_model(dataset_str, model_filename, dataset_config)
        
        nma = NMA(
            loaded_model,
            dataset["X"],
            dataset["y"],
            dataset["labels"],
            graph_type=graph_type,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        
        edges_df_obj = EdgesDataframe(model_filename, dataframe_filename, nma.edges_df)
        edges_df_obj.save_dataframe()
        dendrogram = Dendrogram(dendrogram_filename)
        dendrogram._build_tree_hierarchy(nma.Z, dataset_config["labels"])
        dendrogram.save_dendrogram_as_json(nma.Z)
        
        init_sub_z = dendrogram.get_sub_dendrogram_formatted(nma.Z, dataset_config["init_selected_labels"])
    
        return init_sub_z
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))