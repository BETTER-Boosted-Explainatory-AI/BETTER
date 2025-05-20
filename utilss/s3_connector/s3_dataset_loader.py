import os
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from utilss.s3_connector.s3_handler import S3Handler
from utilss.s3_connector.s3_cifar_loader import S3CifarLoader
from utilss.s3_connector.s3_imagenet_loader import S3ImagenetLoader
from enums.datasets_enum import DatasetsEnum

class S3DatasetLoader:
    def __init__(self, s3_handler=None, bucket_name=None):
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
            
        self.s3_handler = s3_handler or S3Handler(bucket_name=self.bucket_name)
        self.cifar_loader = S3CifarLoader(s3_handler=self.s3_handler)
        self.imagenet_loader = S3ImagenetLoader(s3_handler=self.s3_handler)
    
    def load_from_s3(self, dataset_name):
        """Load dataset from S3 bucket."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            if dataset_name == DatasetsEnum.CIFAR100.value:
                folder_prefix = "cifar100/"
            elif dataset_name == DatasetsEnum.IMAGENET.value:
                folder_prefix = "imagenet/"
            else:
                raise ValueError(f"Unsupported dataset for S3 loading: {dataset_name}")
            
            s3_files = self.s3_handler.list_images(folder_prefix)
            
            if not s3_files:
                print(f"Warning: No files found in S3 bucket with prefix: {folder_prefix}")
            
            for s3_key in s3_files:
                # Keep the relative path structure
                relative_path = s3_key.replace(folder_prefix, '')
                local_path = os.path.join(temp_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                self.s3_handler.download_file(s3_key, local_path)
            
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e
    
    def load_folder(self, dataset_name, folder_type):
        """
        Load a specific folder (clean/adversarial/train)
        """
        if dataset_name == DatasetsEnum.CIFAR100.value:
            return self.cifar_loader.load_cifar100_folder(folder_type)
        elif dataset_name == DatasetsEnum.IMAGENET.value:
            return self.imagenet_loader.load_imagenet_folder(folder_type)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def load_single_image(self, image_key):
        """
        Load a single image from S3
        
        Args:
            image_key (str): S3 key for the image
            
        Returns:
            bytes: Image data
        """
        return self.s3_handler.get_single_image(image_key)
    
    def load_imagenet_train(self):
        return self.imagenet_loader.load_imagenet_train()
    
    def load_cifar100_numpy(self, folder_type):
        """
        Load CIFAR-100 data as NumPy arrays
        
        Args:
            folder_type (str): Type of folder (adversarial or clean)
            
        Returns:
            tuple: (images, labels) as NumPy arrays
        """
        return self.cifar_loader.load_cifar100_as_numpy(folder_type)
    
    def load_cifar100_meta(self):
        return self.cifar_loader.load_cifar100_meta()
    
    def load_dataset_split(self, dataset_name, split_type):
        """
        Load test or train dataset as is
        """
        if split_type not in ['test', 'train']:
            raise ValueError(f"Invalid split type: {split_type}")
        
        return self.load_folder(dataset_name, split_type)
    
    def get_dataset_info(self, dataset_name):
        """Get dataset info directly from S3."""
        try:
            if dataset_name == DatasetsEnum.CIFAR100.value:
                info_file = "cifar100_info.py" 
            elif dataset_name == DatasetsEnum.IMAGENET.value:
                info_file = "imagenet_info.py"
            else:
                raise ValueError(f"Unsupported dataset for S3 info loading: {dataset_name}")
            
            module = self.s3_handler.load_python_module_from_s3(info_file)
            
            if dataset_name == DatasetsEnum.CIFAR100.value:
                return getattr(module, "CIFAR100_INFO")
            elif dataset_name == DatasetsEnum.IMAGENET.value:
                return getattr(module, "IMAGENET_INFO")
                
        except Exception as e:
            print(f"Error loading dataset info from S3: {str(e)}")
            raise
    
    def load_numpy_data(self, dataset_name, folder_type):
        """
        Load NumPy data (.npy files) from clean/adversarial folders
        
        Args:
            dataset_name (str): Name of the dataset (cifar100 or imagenet)
            folder_type (str): Type of folder (clean or adversarial)
            
        Returns:
            dict: Dictionary of NumPy arrays with keys as file names
        """
        if dataset_name == DatasetsEnum.CIFAR100.value:
            return self.cifar_loader.load_cifar100_numpy_files(folder_type)
        elif dataset_name == DatasetsEnum.IMAGENET.value:
            return self.imagenet_loader.load_imagenet_numpy_files(folder_type)
        else:
            raise ValueError(f"Unsupported dataset for numpy loading: {dataset_name}")