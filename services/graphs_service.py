from utilss.classes.preprocessing.prediction_graph import PredictionGraph
from utilss.classes.model import Model
from typing import Dict, Any

def fetch_graph_by_model():
    return None

def _create_graph(dataset_str: str, graph: PredictionGraph, model: Model, 
                        dataset, dataset_config: Dict[str, Any]):
    """Create graph edges based on dataset type."""
    if dataset_str == "imagenet":
        edges_df= graph.create_graph(
            model=model,
            top_k=model.top_k,
            trainset_path=f'{dataset_config["directory_path"]}/train',
            labels_dict=dataset_config["labels_dict"],
        )
        graph.save_graph()
        return edges_df
    elif dataset_str == "cifar100":
        edges_df = graph.create_graph(
            model=model,
            top_k=model.top_k,
            x_train=dataset.x_train,
            y_train_mapped=dataset.y_train_mapped,
        )
        graph.save_graph()
        return edges_df
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")
    
    
def delete_graph():
    return None

