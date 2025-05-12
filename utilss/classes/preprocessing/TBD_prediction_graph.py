from igraph import Graph
import pandas as pd
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utilss.classes.datasets.TBD_cifar100_batch_predictor import Cifar100BatchPredictor
from utilss.classes.datasets.TBD_imagenet_batch_predictor import ImageNetBatchPredictor    
from .graph_builder import GraphBuilder
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import tensorflow as tf

class PredictionGraph:
    def __init__(self, model_filename, graph_filename, graph_type, labels, threshold, infinity, dataset):
        self.graph = Graph(directed=False)
        self.model_filename = model_filename
        self.graph_filename = graph_filename
        self.graph_type = graph_type
        self.graph.add_vertices(labels)
        self.labels = labels
        self.builder = GraphBuilder(graph_type, threshold, infinity)
        self.dataset = dataset

    def get_graph(self):
        return self.graph

    def save_graph(self):
        try:
            if not os.path.exists(self.graph_filename):
                Graph.write_graphml(self.graph, self.graph_filename, "graphml")
                print(f'Graph has been saved: {self.graph_filename}')
            else:
                print(f'File already exists: {self.graph_filename}')
        except Exception as e:
            print(f'Error saving graph file: {str(e)}')

    def load_graph(self):
        try:
            if os.path.exists(self.graph_filename):
                self.graph = Graph.Read_GraphML(self.graph_filename)
                print(f'Graph has been loaded: {self.graph_filename}')
            else:
                print(f'File not found: {self.graph_filename}')
        except Exception as e:
            print(f'Error loading graph file: {str(e)}')
            
    def create_graph(self, model, top_k, trainset_path=None, labels_dict=None, x_train=None, y_train_mapped=None, batch_size=32, min_confidence=0.8):
        """
        Create a graph based on the dataset type.
        
        Args:
            model: The model to use for predictions
            top_k: Number of top predictions to consider
            trainset_path: Path to the ImageNet training set (required for ImageNet)
            labels_dict: Dictionary mapping folder names to labels (required for ImageNet)
            x_train: Training images (required for CIFAR-100)
            y_train_mapped: Training labels (required for CIFAR-100)
            batch_size: Batch size for predictions
            min_confidence: Minimum confidence threshold for adding edges
            
        Returns:
            DataFrame containing edge data
        """
        if self.dataset == "imagenet":
            if trainset_path is None or labels_dict is None:
                raise ValueError("trainset_path and labels_dict are required for ImageNet dataset")
            return self.create_graph_imagenet(model, top_k, trainset_path, labels_dict, batch_size, min_confidence)
        elif self.dataset == "cifar100":
            if x_train is None or y_train_mapped is None:
                raise ValueError("x_train and y_train_mapped are required for CIFAR-100 dataset")
            return self.create_graph_cifar100(model, top_k, x_train, y_train_mapped, batch_size, min_confidence)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def create_graph_cifar100(self, model, top_k, x_train, y_train_mapped, batch_size=64, min_confidence=0.8):
        """
        Create a graph using predictions from the model.
        Optimized for CIFAR-100 data.
        """
        edges_data = []
        batch_images = []
        batch_labels = []

        x_train = preprocess_input(x_train)
        
        total_images = len(x_train)
        print(f"Total CIFAR-100 images to process: {total_images}")
        start_time = time.time()
        
        # Create a predictor if needed
        if not hasattr(model, 'batch_predictor'):
            predictor = Cifar100BatchPredictor(model=model.model, batch_size=batch_size)
        else:
            predictor = model.batch_predictor
            predictor.batch_size = batch_size
        
        for i, image in enumerate(x_train):
            source_label = y_train_mapped[i]
            
            # Add image and label to the batch
            batch_images.append(image)
            batch_labels.append(source_label)
            
            # When the batch is full, or it's the last image, predict the top predictions
            if len(batch_images) == predictor.batch_size or i == len(x_train) - 1:
                # Perform the batch prediction
                top_predictions_batch = predictor.get_top_predictions(batch_images, top_k)
                
                # Iterate over each image in the batch and process predictions
                added_labels = []
                for j, top_predictions in enumerate(top_predictions_batch):
                    current_label = batch_labels[j]
                    
                    # Process the top predictions for each image
                    if top_predictions[0][2] > min_confidence:
                        filtered_predictions = top_predictions[:top_k]
                        
                        # Track which labels were found in predictions
                        
                        for _, pred_label, pred_prob in filtered_predictions:
                            if pred_label not in self.labels:
                                print(f"Warning: Prediction label '{pred_label}' not in graph labels, skipping.")
                                continue
                            
                            # Check if edge should be added
                            if self.builder.should_edge_be_added(current_label, pred_label, pred_prob):
                                # Get edge data and update graph
                                edge_data = self.builder.update_graph(
                                    self.graph, current_label, pred_label, pred_prob, i
                                )
                                edges_data.append(edge_data)
                                added_labels.append(pred_label)
                        
                    # Add infinity edges for categories not in the predictions
                    if self.graph_type == "dissimilarity":
                        for label in self.labels:
                            if label != current_label:  # Don't add self-loops
                                self.builder.add_infinity_edges(
                                    self.graph, added_labels, label, current_label
                                )
            
                # Clear the batch after processing
                batch_images = []
                batch_labels = []
        
        end_time = time.time()
        print(f"Graph creation completed in {end_time - start_time:.2f} seconds")
        
        edges_dataframe = pd.DataFrame(edges_data)
        print(f"Graph creation complete. {len(edges_dataframe)} edges created.")
        return edges_dataframe
    def create_graph_imagenet(self, model, top_k, trainset_path, labels_dict, batch_size=64, min_confidence=0.8):
        """
        Create a graph using predictions from the model.
        Optimized for GPU usage with ImageNet data.
        """
        for label in labels_dict.values():
            if not any(v["name"] == label for v in self.graph.vs):
                self.graph.add_vertices(1)  # Adds a single node
                self.graph.vs[-1]["name"] = label
        
        edges_data = []
        
        folders = os.listdir(trainset_path)
        start_time = time.time()
        
        for folder_name in folders:
            folder_path = os.path.join(trainset_path, folder_name)
        
            if folder_name not in labels_dict:
                print(f"Skipping folder '{folder_name}' - not found in labels_dict")
                continue
            
            image_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]            

            image_label = labels_dict[folder_name]

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_names = [os.path.basename(path) for path in batch_paths]            
                
                if not hasattr(model, 'batch_predictor'):
                    predictor = ImageNetBatchPredictor(model=model.model, batch_size=batch_size)
                else:
                    predictor = model.batch_predictor
                    predictor.batch_size = batch_size 
                    
                batch_predictions = predictor.predict_batch(batch_paths, top_k) 
                
                for idx, (image_path, image_name, predictions_set) in enumerate(zip(batch_paths, batch_names, batch_predictions)):
                    infinity_edges_labels = []
                    
                if predictions_set[0][0] == folder_name:
                    for _, pred_label, pred_prob in predictions_set:    
                        if pred_label not in self.labels:
                            print(f"Warning: Prediction label '{pred_label}' not in graph labels, skipping.")
                            continue
                        
                        if self.builder.should_edge_be_added(image_label, pred_label, pred_prob):                            
                            weight = self.builder.get_edge_weight(pred_prob)
                            if self.builder.should_edge_be_added(image_label, pred_label, weight):
                                edge_data = self.builder.update_graph(
                                    self.graph, image_label, pred_label, pred_prob, image_name
                                )
                                edges_data.append(edge_data)
                                infinity_edges_labels.append(pred_label)
                    
                    if self.graph_type == "dissimilarity":
                        for label in self.labels:
                            self.builder.add_infinity_edges(self.graph, infinity_edges_labels, label, image_label)                        
        
        end_time = time.time()
        print(f"Graph creation completed in {end_time - start_time:.2f} seconds")
        
        edges_dataframe = pd.DataFrame(edges_data)
        print(f"Graph creation complete. {len(edges_dataframe)} edges created.")
        return edges_dataframe            
       