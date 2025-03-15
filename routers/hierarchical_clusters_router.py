from fastapi import APIRouter, Depends, HTTPException, Request
from request_models.hierarchical_clustering_model import HierarchicalClusterRequest
import os
import tensorflow as tf

from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
from utilss.classes.model import Model
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.classes.preprocessing.prediction_graph import PredictionGraph
from utilss.classes.preprocessing.edges_dataframe import EdgesDataframe
from utilss.classes.heap_processor import HeapGraphProcessor
from tensorflow.keras.applications import ResNet50

hierarchical_clusters_router = APIRouter()

@hierarchical_clusters_router.post("/hierarchical_clusters", response_model={})
async def get_hierarchical_clusters(hierarchical_clusters_data: HierarchicalClusterRequest):
    model_filename = hierarchical_clusters_data.model_filename
    graph_type = hierarchical_clusters_data.graph_type
    dataset_str = hierarchical_clusters_data.dataset
    print(f"Model filename: {model_filename}")
    print(f"Graph type: {graph_type}")
    print(f"Dataset: {dataset_str}")
    
    if model_filename is None:
        raise HTTPException(status_code=404, detail="Model filename is required")
    if graph_type != "similarity" and graph_type != "dissimilarity":
        raise HTTPException(status_code=404, detail="Graph type is required")
    
    dataset = None
    dataset_info = {}
    if dataset_str == "cifar100":
        dataset_info = CIFAR100_INFO
        dataset = DatasetFactory.create_dataset(CIFAR100_INFO["dataset"])
        dataset.load(CIFAR100_INFO["dataset"])
    elif dataset_str == "imagenet":
        dataset_info = IMAGENET_INFO
        dataset = DatasetFactory.create_dataset(IMAGENET_INFO["dataset"])
        dataset.load(IMAGENET_INFO["dataset"])
    else:
        raise HTTPException(status_code=404, detail=f'Invalid dataset: {dataset_str} ')
    
    try:
        # graph_types = ["similarity", "dissimilarity", "confusion_matrix"]
        heap_type = ["min", "max"]
        
        weights = "imagenet"
        # original_model_filename = "data/database/models/cifar100_resnet.keras"

        # model_filename = f'data/graphs/resnet50_mini_imagenet.keras'
        model_path = f'data/database/models/{model_filename}.keras'
        dataframe_filename = f'data/graphs/edges_{graph_type}_{model_filename}.csv'
        graph_filename = f'data/graphs/graph_{graph_type}_{model_filename}.graphml'
        
        if dataset_str == "imagenet":
            resnet_model = Model(
            ResNet50(weights=weights), dataset_info["top_k"], dataset_info["min_confidence"], model_path, dataset_info["dataset"]
            )
        
            new_graph = PredictionGraph(model_path, graph_filename, graph_type, dataset_info["labels"], dataset_info["threshold"], dataset_info["infinity"], dataset_info["dataset"])
            edges_df_temp = new_graph.create_graph(
                model=resnet_model,
                top_k=resnet_model.top_k,
                trainset_path=f'{dataset_info["directory_path"]}/train',  # ImageNet-specific
                labels_dict=dataset_info["labels_dict"],      # ImageNet-specific
            )
        elif dataset_str == "cifar100":
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f'Model {model_path} does not exist')
        
            resnet_model_cifar100 = tf.keras.models.load_model(model_path)
            print(f"Model {model_path} has been loaded")
            resnet_model = Model(
                resnet_model_cifar100, dataset_info["top_k"], dataset_info["min_confidence"], model_path, dataset_info["dataset"]
            )
            
            new_graph = PredictionGraph(model_path, graph_filename, graph_type, dataset_info["labels"], dataset_info["threshold"], dataset_info["infinity"], dataset_info["dataset"])            
            edges_df_temp = new_graph.create_graph(
                model=resnet_model,
                top_k=resnet_model.top_k,
                x_train=dataset.x_train,              # CIFAR-100 specific
                y_train_mapped=dataset.y_train_mapped,# CIFAR-100 specific
            )    
            
        edges_df = EdgesDataframe(resnet_model, dataframe_filename, edges_df_temp)
        new_graph.save_graph()
        edges_df.save_dataframe()
    except Exception as e:
        raise HTTPException(status_code=417, detail=f'Expectation Failed: {str(e)}')
    
    return {"message": "Graph created successfully"}
