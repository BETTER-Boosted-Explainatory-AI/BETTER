from fastapi import APIRouter, Depends, HTTPException, Request, status
from .dataset_service import _get_dataset_config, _load_dataset
from .graphs_service import _create_graph
from .models_service import _load_model
from .union_find_service import _create_uf
from .heap_service import _create_graph_heap, _create_matrix_heap
from .confusion_matrix_service import _create_confusion_matrix
from utilss.classes.preprocessing.edges_dataframe import EdgesDataframe
from utilss.classes.preprocessing.prediction_graph import PredictionGraph
from utilss.classes.union_find import UnionFind
from utilss.classes.hierarchical_cluster import HierarchicalCluster

import numpy as np

def get_hierarchical_cluster_by_model():
    return None 

def post_hierarchical_cluster(model_filename, graph_type, dataset_str):
    if model_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
        
    if graph_type not in ["similarity", "dissimilarity"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity' or 'dissimilarity'"
        )
    
    # Get dataset configuration
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        model_path = f'data/database/models/{model_filename}.keras'
        dataframe_filename = f'data/database/dataframes/edges_{graph_type}_{model_filename}.csv'
        graph_filename = f'data/database/graphs/graph_{graph_type}_{model_filename}.graphml'
        
        loaded_model = _load_model(dataset_str, model_path, dataset_config)
        
        new_graph = PredictionGraph(
            model_path, 
            graph_filename, 
            graph_type, 
            dataset_config["labels"], 
            dataset_config["threshold"], 
            dataset_config["infinity"], 
            dataset_config["dataset"]
        )
        edges_df = _create_graph(dataset_str, new_graph, loaded_model, dataset, dataset_config)
        
        edges_df_obj = EdgesDataframe(model_path, dataframe_filename, edges_df)
        edges_df_obj.save_dataframe()
        
        # normilize edges
        for edge in new_graph.get_graph().es:
            edge["weight"] = np.log1p(edge["weight"])
            
        heap_type = "max" if graph_type == "similarity" else "min"
        
        new_heap = _create_graph_heap(new_graph.get_graph(), heap_type, dataset_config["labels"])
        uf, merge_list = _create_uf(new_heap, dataset_config["labels"], heap_type)
        
        labels_dict = None
        if dataset_str == "imagenet":
            labels_dict = dataset_config["labels_dict"]
        
        hc = HierarchicalCluster(labels_dict)
        hc.create_dendrogram_data(uf, dataset_config["labels"], uf.max_weight)
        hc.save_dendrogram_as_json(dataset_config["labels"], f'data/database/dendrograms/dendrogram_{graph_type}_{model_filename}.json')
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

def delete_hierarchical_cluster():
    return None

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
        model_path = f'data/database/models/{model_filename}.keras'
        dataframe_filename = f'{edges_df_filename}.csv'
        
        dataset_config = _get_dataset_config(dataset_str)
        
        edges_df_obj = EdgesDataframe(model_path, dataframe_filename)
        edges_df_obj.load_dataframe()
        count_df = edges_df_obj.get_dataframe_by_count()
        print(count_df)
        
        confusion_matrix = _create_confusion_matrix(count_df, dataset_config["labels"])
        
        confusion_matrix_t = confusion_matrix + confusion_matrix.T
        max_value = np.max(confusion_matrix_t)
        confusion_matrix_t = max_value - confusion_matrix_t
               
        heap_type = "min"
        new_heap = _create_matrix_heap(confusion_matrix_t, heap_type, dataset_config["labels"])
        uf, merge_list = _create_uf(new_heap, dataset_config["labels"], heap_type)

        if dataset_str == "imagenet":
            labels_dict = dataset_config["labels_dict"]
        else:
            labels_dict = None
        
        hc = HierarchicalCluster(labels_dict)
        hc.create_dendrogram_data(uf, dataset_config["labels"], uf.max_weight)
        hc.save_dendrogram_as_json(dataset_config["labels"], f'data/database/dendrograms/dendrogram_confusion_matrix_{model_filename}.json')
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
        
