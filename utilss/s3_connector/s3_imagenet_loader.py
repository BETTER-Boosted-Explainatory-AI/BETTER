import os
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Any
from utilss.s3_connector.s3_handler import S3Handler

class S3ImagenetLoader:

    def __init__(self, s3_handler=None, bucket_name=None):
        """Initialize with an S3 handler."""
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
            
        self.s3_handler = s3_handler or S3Handler(bucket_name=self.bucket_name)
    
    def load_imagenet_folder(self, folder_type: str) -> str:
        temp_dir = tempfile.mkdtemp()
        
        try:
            files = self.s3_handler.get_folder_contents('imagenet', folder_type)
            
            if not files:
                print(f"Warning: No files found in folder: imagenet/{folder_type}")
                return temp_dir
            
            # Download all files to temp directory
            for s3_key in files:
                relative_path = s3_key.replace(f"imagenet/{folder_type}/", '')
                local_path = os.path.join(temp_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3_handler.download_file(s3_key, local_path)
            
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e
    
    
    def load_imagenet_train(self) -> str:
        temp_dir = tempfile.mkdtemp()
        try:
            self.s3_handler.get_imagenet_train_with_subdirs(temp_dir)
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e
    
    def load_imagenet_numpy_files(self, folder_type: str) -> Dict[str, np.ndarray]:
        if folder_type not in ['clean', 'adversarial']:
            raise ValueError(f"Invalid folder type for numpy data: {folder_type}")
            
        return self.s3_handler.get_numpy_data('imagenet', folder_type)