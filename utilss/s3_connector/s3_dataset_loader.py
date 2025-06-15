import os
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
from utilss.s3_connector.s3_handler import S3Handler
from utilss.s3_connector.s3_cifar_loader import S3CifarLoader
from utilss.s3_connector.s3_imagenet_loader import S3ImagenetLoader
from utilss.enums.datasets_enum import DatasetsEnum

class S3DatasetLoader:
    def __init__(self, s3_handler=None, bucket_name=None):
        self.bucket_name = bucket_name or os.environ.get('S3_DATASETS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
            
        self.s3_handler = s3_handler or S3Handler(bucket_name=self.bucket_name)
        self.cifar_loader = S3CifarLoader(s3_handler=self.s3_handler)
        self.imagenet_loader = S3ImagenetLoader(s3_handler=self.s3_handler)
    
    
    def load_folder(self, dataset_name, folder_type):
        """List files in a specific dataset folder"""
        if dataset_name == DatasetsEnum.CIFAR100.value:
            return self.cifar_loader.list_cifar100_files(folder_type)
        elif dataset_name == DatasetsEnum.IMAGENET.value:
            return self.imagenet_loader.load_imagenet_folder(folder_type)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def load_single_image(self, image_key):
        """Get a single image from S3"""
        return self.s3_handler.get_single_image(image_key)
    
    def get_image_stream(self, image_key):
        """Get image as a stream for processing without downloading"""
        # Make sure to handle method name changes
        if hasattr(self.s3_handler, 'get_image_stream'):
            return self.s3_handler.get_image_stream(image_key)
        elif hasattr(self.s3_handler, 'get_object_stream'):
            return self.s3_handler.get_object_stream(image_key)
        else:
            raise AttributeError("S3Handler has no stream method available")
    
    def load_imagenet_train(self):
        """List ImageNet training classes and images"""
        return self.imagenet_loader.get_imagenet_classes()
    
    def load_imagenet_test(self) -> List[str]:
        """Shortcut: all test images for ImageNet."""
        return self.imagenet_loader.list_test_images()
    
    def load_cifar100_numpy(self, folder_type):
        """Load CIFAR-100 dataset as numpy arrays"""
        return self.cifar_loader.load_cifar100_as_numpy(folder_type)
    
    def load_cifar100_meta(self):
        """Load CIFAR-100 metadata"""
        return self.cifar_loader.load_cifar100_meta()
    
    
    def get_dataset_info(self, dataset_name):
        """Get dataset info directly from S3"""
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
        if dataset_name == DatasetsEnum.CIFAR100.value:
            return self.cifar_loader.load_cifar100_numpy_files(folder_type)
        elif dataset_name == DatasetsEnum.IMAGENET.value:
            return self.imagenet_loader.load_imagenet_numpy_files(folder_type)
        else:
            raise ValueError(f"Unsupported dataset for numpy loading: {dataset_name}")