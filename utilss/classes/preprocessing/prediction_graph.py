from igraph import Graph
import pandas as pd
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from classes.datasets.cifar100_batch_predictor import Cifar100BatchPredictor
from classes.datasets.imagenet_batch_predictor import ImageNetBatchPredictor    


class PredictionGraph:
    def __init__(self, model_filename, graph_filename, graph_type, labels):
        """Initialize a prediction graph.
        
        Args:
            filename: Path to save/load the graph
            graph_type: Type of graph ("cifar100" or "imagenet")
            labels: List of class labels
        """
        self.graph = Graph(directed=False)
        self.model_filename = model_filename
        self.graph_filename = graph_filename
        self.graph_type = graph_type
        self.graph.add_vertices(labels)
        self.labels = labels

    def get_graph(self):
        """Get the current graph."""
        return self.graph

    def save_graph(self):
        """Save the graph to a file."""
        if not os.path.exists(self.graph_filename):
            Graph.write_graphml(self.graph, self.graph_filename, "graphml")
            print(f'Graph has been saved: {self.graph_filename}')

    def load_graph(self):
        """Load a graph from a file."""
        if os.path.exists(self.graph_filename):
            self.graph = Graph.Read_GraphML(self.graph_filename)
            print(f'Graph has been loaded: {self.graph_filename}')

    def create_graph(self, model, top_k, threshold, infinity, percent=0.8, batch_size=64, **kwargs):
        """Create a graph using predictions from the model.
        
        This is a facade method that delegates to the appropriate implementation
        based on the graph type.
        
        Args:
            model: The prediction model
            top_k: Number of top predictions to consider
            threshold: Minimum probability threshold
            infinity: Weight for missing connections
            percent: Confidence threshold for predictions
            batch_size: Size of batches for prediction
            **kwargs: Additional arguments specific to each graph type
        
        Returns:
            DataFrame containing the edge data
        """
        if self.graph_type == "cifar100":
            return self._create_cifar100_graph(
                model=model,
                top_k=top_k,
                threshold=threshold,
                infinity=infinity,
                x_train=kwargs.get('x_train'),
                y_train=kwargs.get('y_train'),
                class_names=self.labels,
                percent=percent,
                batch_size=batch_size
            )
        elif self.graph_type == "imagenet":
            return self._create_imagenet_graph(
                model=model,
                top_k=top_k,
                threshold=threshold,
                infinity=infinity,
                labels_dict=kwargs.get('labels_dict'),
                trainset_path=kwargs.get('trainset_path'),
                percent=percent,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")