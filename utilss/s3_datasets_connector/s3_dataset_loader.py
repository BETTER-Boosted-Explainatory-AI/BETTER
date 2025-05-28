import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tempfile
import shutil
from utilss.enums.datasets_enum import DatasetsEnum
from s3_handler import S3Handler


class S3DatasetLoader:
    """Class for loading datasets from S3."""
    
    def __init__(self, s3_handler=None):
        """Initialize with an S3 handler."""
        self.s3_handler = s3_handler or S3Handler()
    

    def load_from_s3(self, dataset_name):
        """Load dataset from S3 bucket."""
        temp_dir = tempfile.mkdtemp()
        try:
            if dataset_name == DatasetsEnum.CIFAR100.value:
                s3_prefix = "cifar100/"
                info_file = "cifar100_info.py"
            elif dataset_name == DatasetsEnum.IMAGENET.value:
                s3_prefix = "imagenet/"
                info_file = "imagenet_info.py"
            else:
                raise ValueError(f"Unsupported dataset for S3 loading: {dataset_name}")
            
            s3_files = self.s3_handler.list_objects(s3_prefix)
            
            if not s3_files:
                print(f"Warning: No files found in S3 bucket with prefix: {s3_prefix}")
            
            for s3_key in s3_files:
                relative_path = s3_key[len(s3_prefix):]
                local_path = os.path.join(temp_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                self.s3_handler.download_file(s3_key, local_path)
            
            try:
                info_local_path = os.path.join(temp_dir, os.path.basename(info_file))
                self.s3_handler.download_file(info_file, info_local_path)
            except Exception as e:
                print(f"Warning: Could not download info file {info_file}: {str(e)}")
            
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e
