import boto3
import os
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.enums.datasets_enum import DatasetsEnum
from utilss.s3_datasets_connector.s3_handler import S3Handler

class S3DatasetLoader:
    """Class for loading datasets from S3."""
    
    def __init__(self, s3_handler=None):
        """Initialize with an S3 handler."""
        self.s3_handler = s3_handler or S3Handler()
    
    def load_from_s3(self, dataset_name):
        """Load dataset from S3 bucket to a temporary directory."""
        # Create a temporary directory to store the downloaded files
        temp_dir = tempfile.mkdtemp()
        try:
            # Determine the S3 prefix based on the dataset name
            if dataset_name == DatasetsEnum.CIFAR100.value:
                s3_prefix = "data/datasets/cifar100/"
            elif dataset_name == DatasetsEnum.IMAGENET.value:
                s3_prefix = "data/datasets/imagenet/"
            else:
                raise ValueError(f"Unsupported dataset for S3 loading: {dataset_name}")
            
            # List all relevant files in the S3 bucket
            s3_files = self.s3_handler.list_objects(s3_prefix)
            
            if not s3_files:
                raise FileNotFoundError(f"No files found in S3 bucket with prefix: {s3_prefix}")
            
            # Download all files to the temp directory
            for s3_key in s3_files:
                # Preserve the directory structure relative to the dataset
                relative_path = s3_key[len(s3_prefix):]
                local_path = os.path.join(temp_dir, relative_path)
                self.s3_handler.download_file(s3_key, local_path)
            
            # Return the path to the temp directory containing the dataset
            return temp_dir
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir)
            raise e
