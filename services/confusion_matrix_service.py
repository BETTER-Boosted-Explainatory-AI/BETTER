import numpy as np
from fastapi import HTTPException, status
from .dataset_service import _get_dataset_config, _load_dataset
from .union_find_service import _create_uf
from .heap_service import _create_matrix_heap
from utilss.classes.preprocessing.edges_dataframe import EdgesDataframe
from utilss.classes.hierarchical_cluster import HierarchicalCluster
from utilss.enums.datasets_enum import DatasetsEnum
from utilss.enums.heap_type import HeapType

from utilss.classes.dendrogram import Dendrogram
import os

def _create_confusion_matrix(count_per_target, labels):

    labels = sorted(labels)  # Ensure sorted order
    class_index = {cls: i for i, cls in enumerate(labels)}  # Map class names to indices

    # Step 1: Initialize a zero matrix
    confusion_matrix = np.zeros((len(labels), len(labels)))

    # Step 3: Populate only top 5 misclassifications per source
    for source, group in count_per_target.groupby('source'):
        top_5 = group.nlargest(5, 'count')  # Get the top 5 misclassifications for this source

        for _, row in top_5.iterrows():
            target, count = row['target'], row['count']
            if source in class_index and target in class_index:  # Ensure both exist in class_names
                i, j = class_index[source], class_index[target]
                confusion_matrix[i, j] = count

    return confusion_matrix

def post_hierarchical_cluster_confusion_matrix(model_filename, edges_df_filename, dataset_str):
    if model_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
    
    if edges_df_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Edges Dataframe filename is required"
        )
            
    try:
        dataframe_filename = f'{edges_df_filename}'
        dendrogram_filename = f'dendrogram_confusion_matrix_{model_filename}'
        
        dataset_config = _get_dataset_config(dataset_str)
        
        edges_df_obj = EdgesDataframe(model_filename, dataframe_filename)
        edges_df_obj.load_dataframe()
        count_df = edges_df_obj.get_dataframe_by_count()
        
        confusion_matrix = _create_confusion_matrix(count_df, dataset_config["labels"])
        
        confusion_matrix_t = confusion_matrix + confusion_matrix.T
        max_value = np.max(confusion_matrix_t)
        confusion_matrix_t = max_value - confusion_matrix_t
               
        heap_type = HeapType.MINIMUM.value
        new_heap = _create_matrix_heap(confusion_matrix_t, heap_type, dataset_config["labels"])
        uf, merge_list = _create_uf(new_heap, dataset_config["labels"], heap_type)

        if dataset_str == DatasetsEnum.IMAGENET.value:
            labels_dict = dataset_config["labels_dict"]
        else:
            labels_dict = None
        
        hc = HierarchicalCluster(labels_dict)
        hc.create_dendrogram_data(uf, dataset_config["labels"], uf.max_weight)
        dendrogram = Dendrogram(dendrogram_filename)
        dendrogram._build_tree_hierarchy(hc.Z, dataset_config["labels"])
        dendrogram.save_dendrogram_as_json(hc.Z)

        return hc.Z
    
    
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Resource not found: {str(e)}"
        )
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"Invalid parameter: {str(e)}"
        )
    except Exception as e:
        print(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error processing request: {str(e)}"
        )
