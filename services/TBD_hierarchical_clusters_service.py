from fastapi import HTTPException, status
from .dataset_service import _get_dataset_config, _load_dataset
from .TBD_graphs_service import _create_graph
from .models_service import _load_model, construct_model
from .TBD_union_find_service import _create_uf
from .heap_service import _create_graph_heap, _get_heap_type
from utilss.classes.edges_dataframe import EdgesDataframe
from utilss.classes.preprocessing.TBD_prediction_graph import PredictionGraph
from utilss.classes.TBD_hierarchical_cluster import HierarchicalCluster
from utilss.classes.dendrogram import Dendrogram
from utilss.enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.enums.datasets_enum import DatasetsEnum
import numpy as np

def get_hierarchical_cluster_by_model():
    return None 

def post_hierarchical_cluster(model_filename, graph_type, dataset_str):
    if model_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
        
    if graph_type != HierarchicalClusterType.SIMILARITY.value and graph_type != HierarchicalClusterType.DISSIMILARITY.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity' or 'dissimilarity'"
        )
    
    # Get dataset configuration
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        dataframe_filename = f'edges_{graph_type}_{model_filename}.csv'
        graph_filename = f'graph_{graph_type}_{model_filename}.graphml'
        dendrogram_filename = f'dendrogram_{graph_type}_{model_filename}.json'

        loaded_model = _load_model(dataset_str, model_filename, dataset_config)
        
        new_graph = PredictionGraph(
            model_filename, 
            graph_filename, 
            graph_type, 
            dataset_config["labels"], 
            dataset_config["threshold"], 
            dataset_config["infinity"], 
            dataset_config["dataset"]
        )
        edges_df = _create_graph(dataset_str, new_graph, loaded_model, dataset, dataset_config)
        
        edges_df_obj = EdgesDataframe(model_filename, dataframe_filename, edges_df)
        edges_df_obj.save_dataframe()
        
        # normilize edges
        ## add sigmoid normalization for similarity for imagenet 
        for edge in new_graph.get_graph().es:
            edge["weight"] = np.log1p(edge["weight"])
            
        heap_type = _get_heap_type(graph_type)
        
        new_heap = _create_graph_heap(new_graph.get_graph(), heap_type, dataset_config["labels"])
        uf, merge_list = _create_uf(new_heap, dataset_config["labels"], heap_type)
        
        labels_dict = None
        if dataset_str == DatasetsEnum.IMAGENET.value:
            labels_dict = dataset_config["labels_dict"]
        
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
    
def post_new_hierarchical_cluster(model_filename, graph_type, dataset_str, user_id):
    if model_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
        
    if graph_type != HierarchicalClusterType.SIMILARITY.value and graph_type != HierarchicalClusterType.DISSIMILARITY.value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Graph type must be either 'similarity' or 'dissimilarity'"
        )
    
    # Get dataset configuration
    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)
    
    try:
        dataframe_filename = f'edges_{graph_type}_{model_filename}.csv'
        graph_filename = f'graph_{graph_type}_{model_filename}.graphml'
        dendrogram_filename = f'dendrogram_{graph_type}_{model_filename}.json'

        loaded_model = construct_model(model_filename, dataset_config)
        
        new_graph = PredictionGraph(
            model_filename, 
            graph_filename, 
            graph_type, 
            dataset_config["labels"], 
            dataset_config["threshold"], 
            dataset_config["infinity"], 
            dataset_config["dataset"]
        )
        edges_df = _create_graph(dataset_str, new_graph, loaded_model, dataset, dataset_config)
        
        edges_df_obj = EdgesDataframe(model_filename, dataframe_filename, edges_df)
        edges_df_obj.save_dataframe()
        
        # normilize edges
        for edge in new_graph.get_graph().es:
            edge["weight"] = np.log1p(edge["weight"])
            
        heap_type = _get_heap_type(graph_type)
        
        new_heap = _create_graph_heap(new_graph.get_graph(), heap_type, dataset_config["labels"])
        uf, merge_list = _create_uf(new_heap, dataset_config["labels"], heap_type)
        
        labels_dict = None
        if dataset_str == DatasetsEnum.IMAGENET.value:
            labels_dict = dataset_config["labels_dict"]
        
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
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error processing request: {str(e)}"
        )

def delete_hierarchical_cluster():
    return None
