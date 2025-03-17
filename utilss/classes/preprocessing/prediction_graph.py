from igraph import Graph
import pandas as pd
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utilss.classes.datasets.cifar100_batch_predictor import Cifar100BatchPredictor
from utilss.classes.datasets.imagenet_batch_predictor import ImageNetBatchPredictor    
from .graph_builder import GraphBuilder
from tensorflow.keras.applications.resnet50 import preprocess_input
import time

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
            
    def create_graph(self, model, top_k, trainset_path=None, labels_dict=None, x_train=None, y_train_mapped=None, batch_size=64, min_confidence=0.8):
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

    def create_graph_imagenet(self, model, top_k, trainset_path, labels_dict, batch_size=64, min_confidence=0.8):
        """
        Create a graph using predictions from the model.
        Optimized for GPU usage with ImageNet data.
        """
        edges_data = []
        processed_count = 0
        
        # Collect all image paths and their labels
        all_image_paths = []
        all_image_labels = []
        folder_to_images = {}
        
        # First, collect all the image paths to process
        print("Collecting image paths...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Create a list of folder processing tasks
            folder_tasks = []
            
            for folder_name in os.listdir(trainset_path):
                folder_path = os.path.join(trainset_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                    
                # Get label from folder name
                image_label = labels_dict.get(folder_name)
                if image_label is None:
                    continue
                
                folder_tasks.append((folder_name, folder_path, image_label))
            
            # Process folders in parallel to collect image paths
            def process_folder(task):
                folder_name, folder_path, image_label = task
                folder_images = []
                
                for image_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, image_name)
                    if os.path.isfile(img_path):
                        folder_images.append((img_path, image_name))
                
                return folder_name, image_label, folder_images
            
            # Execute all folder processing tasks
            results = list(executor.map(process_folder, folder_tasks))
            
            # Combine results
            for folder_name, image_label, folder_images in results:
                if folder_images:
                    folder_to_images[folder_name] = folder_images
                    for img_path, _ in folder_images:
                        all_image_paths.append(img_path)
                        all_image_labels.append(image_label)
        
        total_images = len(all_image_paths)
        print(f"Total images to process: {total_images}")
        start_time = time.time()

        # Create a BatchPredictor if model is not already one
        if not hasattr(model, 'batch_predictor'):
            predictor = ImageNetBatchPredictor(model=model.model, batch_size=batch_size)
        else:
            predictor = model.batch_predictor
            predictor.batch_size = batch_size  # Ensure correct batch size
        
        # Process images in batches by folder to maintain organization
        for folder_name, image_list in folder_to_images.items():
            if not image_list:
                continue
                
            image_label = labels_dict[folder_name]
            folder_paths = [item[0] for item in image_list]
            folder_names = [item[1] for item in image_list]
            
            # Process in larger batches for better GPU utilization
            for i in range(0, len(folder_paths), batch_size):
                # Get current batch
                batch_paths = folder_paths[i:i+batch_size]
                batch_names = folder_names[i:i+batch_size]
                
                # Get predictions for this batch
                batch_results = predictor.get_top_predictions(batch_paths, top_k)
                
                # Process results and update graph
                for j, (img_path, predictions) in enumerate(batch_results):
                    # Match the image path back to its name
                    # Find the index of img_path in batch_paths
                    try:
                        path_index = batch_paths.index(img_path)
                        image_name = batch_names[path_index]
                    except ValueError:
                        # If path not found, use a default approach
                        image_name = os.path.basename(img_path)
                    
                    # Skip if no predictions or confidence too low
                    if not predictions or predictions[0][2] < min_confidence:
                        continue
                    
                    # Only process images predicted correctly
                    correct_prediction = False
                    for _, pred_label, _ in predictions:
                        if pred_label == image_label:
                            correct_prediction = True
                            break
                            
                    if not correct_prediction:
                        continue
                        
                    # Process predictions and update graph
                    infinity_edges_labels = []
                    for _, pred_label, pred_prob in predictions:
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
                
                processed_count += len(batch_paths)
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count}/{total_images} images")
        
        end_time = time.time()
        print(f"Graph creation completed in {end_time - start_time:.2f} seconds")
        
        edges_dataframe = pd.DataFrame(edges_data)
        print(f"Graph creation complete. {len(edges_dataframe)} edges created.")
        return edges_dataframe
    
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

    def create_graph_imagenet(self, model, top_k, trainset_path, labels_dict, batch_size=64, min_confidence=0.8):
        """
        Create a graph using predictions from the model.
        Optimized for GPU usage with ImageNet data.
        """
        edges_data = []
        processed_count = 0
        
        # Collect all image paths and their labels
        all_image_paths = []
        all_image_labels = []
        folder_to_images = {}
        
        # First, collect all the image paths to process
        print("Collecting image paths...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Create a list of folder processing tasks
            folder_tasks = []
            
            for folder_name in os.listdir(trainset_path):
                folder_path = os.path.join(trainset_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                    
                # Get label from folder name
                image_label = labels_dict.get(folder_name)
                if image_label is None:
                    continue
                
                folder_tasks.append((folder_name, folder_path, image_label))
            
            # Process folders in parallel to collect image paths
            def process_folder(task):
                folder_name, folder_path, image_label = task
                folder_images = []
                
                for image_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, image_name)
                    if os.path.isfile(img_path):
                        folder_images.append((img_path, image_name))
                
                return folder_name, image_label, folder_images
            
            # Execute all folder processing tasks
            results = list(executor.map(process_folder, folder_tasks))
            
            # Combine results
            for folder_name, image_label, folder_images in results:
                if folder_images:
                    folder_to_images[folder_name] = folder_images
                    for img_path, _ in folder_images:
                        all_image_paths.append(img_path)
                        all_image_labels.append(image_label)
        
        total_images = len(all_image_paths)
        print(f"Total images to process: {total_images}")
        start_time = time.time()

        # Create a BatchPredictor if model is not already one
        if not hasattr(model, 'batch_predictor'):
            predictor = ImageNetBatchPredictor(model=model.model, batch_size=batch_size)
        else:
            predictor = model.batch_predictor
            predictor.batch_size = batch_size  # Ensure correct batch size
        
        # Process images in batches by folder to maintain organization
        for folder_name, image_list in folder_to_images.items():
            if not image_list:
                continue
                
            image_label = labels_dict[folder_name]
            folder_paths = [item[0] for item in image_list]
            folder_names = [item[1] for item in image_list]
            
            # Process in larger batches for better GPU utilization
            for i in range(0, len(folder_paths), batch_size):
                # Get current batch
                batch_paths = folder_paths[i:i+batch_size]
                batch_names = folder_names[i:i+batch_size]
                
                # Get predictions for this batch
                batch_results = predictor.get_top_predictions(batch_paths, top_k)
                
                # Process results and update graph
                for j, (img_path, predictions) in enumerate(batch_results):
                    # Match the image path back to its name
                    # Find the index of img_path in batch_paths
                    try:
                        path_index = batch_paths.index(img_path)
                        image_name = batch_names[path_index]
                    except ValueError:
                        # If path not found, use a default approach
                        image_name = os.path.basename(img_path)
                    
                    # Skip if no predictions or confidence too low
                    if not predictions or predictions[0][2] < min_confidence:
                        continue
                    
                    # Only process images predicted correctly
                    correct_prediction = False
                    for _, pred_label, _ in predictions:
                        if pred_label == image_label:
                            correct_prediction = True
                            break
                            
                    if not correct_prediction:
                        continue
                        
                    # Process predictions and update graph
                    infinity_edges_labels = []
                    for _, pred_label, pred_prob in predictions:
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
                
                processed_count += len(batch_paths)
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count}/{total_images} images")
        
        end_time = time.time()
        print(f"Graph creation completed in {end_time - start_time:.2f} seconds")
        
        edges_dataframe = pd.DataFrame(edges_data)
        print(f"Graph creation complete. {len(edges_dataframe)} edges created.")
        return edges_dataframe
        
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