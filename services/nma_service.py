from fastapi import HTTPException, status
import os
from .dataset_service import _get_dataset_config, _load_dataset
from .models_service import _load_model, _get_model_path
from utilss.classes.nma import NMA
from utilss.classes.edges_dataframe import EdgesDataframe
from utilss.classes.dendrogram import Dendrogram
from utilss.enums.datasets_enum import DatasetsEnum
from utilss.enums.graph_types import GraphTypes

def _create_nma(model_file, graph_type, dataset_str, user_id, min_confidence, top_k):
    if model_file is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model file is required"
        )
        
    if graph_type != GraphTypes.SIMILARITY.value and graph_type != GraphTypes.DISSIMILARITY.value and graph_type != GraphTypes.COUNT.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity' or 'dissimilarity'"
        )
    
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        loaded_model = _load_model(dataset_str, model_file.name, dataset_config)
        model_directory = _get_model_path(user_id, loaded_model.model_path)
        if model_directory is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Could not find model directory"
            )
            
        dataframe_filename = f'{model_directory}/{graph_type}/edges_df.csv'
        dendrogram_filename = f'{model_directory}/{graph_type}/dendrogram'
        
        labels = dataset.labels
        if dataset_str == DatasetsEnum.IMAGENET.value:
            labels = dataset.directory_labels

        nma = NMA(
            loaded_model.model,
            dataset.x_train,
            dataset.y_train,
            labels,
            graph_type=graph_type,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        
        edges_df_obj = EdgesDataframe(model_file.name, dataframe_filename, nma.edges_df)
        edges_df_obj.save_dataframe()
        print(nma.Z)            
        
        dendrogram = Dendrogram(dendrogram_filename, nma.Z)
        dendrogram._build_tree_hierarchy(nma.Z, dataset.labels)
        
        dendrogram.save_dendrogram(nma.Z)
        
        init_sub_z = dendrogram.get_sub_dendrogram_formatted(dataset_config["init_selected_labels"])
        print(init_sub_z)
        return init_sub_z
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))