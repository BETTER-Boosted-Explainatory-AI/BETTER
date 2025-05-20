import boto3
import os
import importlib.util
import numpy as np
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile
import pickle
import tempfile
import io


class S3Handler:
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, bucket_name=None):
        """Initialize S3 handler with AWS credentials."""
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials not found")
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name not specified")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
    
    def download_file(self, s3_key, local_path):
        """Download a file from S3 to a local path."""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except (NoCredentialsError, ClientError) as e:
            print(f"Error downloading file from S3: {str(e)}")
            return False
    
    def list_images(self, prefix=''):
        """List objects in the S3 bucket with the given prefix."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            
            return objects
        except (NoCredentialsError, ClientError) as e:
            print(f"Error listing S3 objects: {str(e)}")
            return []
    
    
    def get_folder_contents(self, dataset_name, folder_type):
        """
        Return contents of a specific folder (clean/adversarial/train)
        """
        prefix = f"{dataset_name}/{folder_type}/"
        return self.list_images(prefix)
    
    def get_single_image(self, image_key):
        """
        Return a single image from S3
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            return response['Body'].read()
        except (NoCredentialsError, ClientError) as e:
            print(f"Error getting image from S3: {str(e)}")
            return None
    
    def get_imagenet_train_with_subdirs(self, local_dir):
        """
        Download ImageNet training data preserving subdirectory structure
        """
        prefix = "imagenet/train/"
        s3_files = self.list_images(prefix)
        
        for s3_key in s3_files:
            # Keep the subdirectory structure
            relative_path = s3_key.replace(prefix, '')
            local_path = os.path.join(local_dir, relative_path)
            
            # Create directories as needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            self.download_file(s3_key, local_path)
        
        return local_dir
    
    def get_cifar100_as_numpy(self, folder_type):
        prefix = f"cifar100/{folder_type}/"
        s3_files = self.list_images(prefix)
        
        # CIFAR-100 data is typically stored in binary format
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Download all files first
            for s3_key in s3_files:
                local_path = os.path.join(temp_dir, os.path.basename(s3_key))
                self.download_file(s3_key, local_path)
            
            # Process files into NumPy arrays
            # Assuming the format is similar to the official CIFAR-100 format
            images = []
            labels = []
            
            for filename in os.listdir(temp_dir):
                if filename.endswith('.bin') or filename.endswith('.pkl'):
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f, encoding='bytes')
                        if b'data' in data:
                            batch_images = data[b'data']
                            batch_labels = data[b'fine_labels'] if b'fine_labels' in data else data[b'labels']
                            
                            images.append(batch_images)
                            labels.append(batch_labels)
            
            if images:
                images = np.vstack(images)
                # Reshape to [N, 3, 32, 32] and then transpose to [N, 32, 32, 3]
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels = np.concatenate(labels)
                
            return images, labels
        
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def get_cifar100_meta(self):
        meta_key = "cifar100/meta"
        
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "meta")
        
        try:
            if self.download_file(meta_key, local_path):
                with open(local_path, 'rb') as f:
                    meta_data = pickle.load(f, encoding='bytes')
                return meta_data
            return None
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def get_dataset_split(self, dataset_name, split_type):
        """
        Get test or train dataset as is
        """
        prefix = f"{dataset_name}/{split_type}/"
        temp_dir = tempfile.mkdtemp()
        
        s3_files = self.list_images(prefix)
        
        for s3_key in s3_files:
            # Keep the relative path structure
            relative_path = s3_key.replace(prefix, '')
            local_path = os.path.join(temp_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.download_file(s3_key, local_path)
        
        return temp_dir
    
    def load_python_module_from_s3(self, s3_key):
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_path = temp_file.name
            
            success = self.download_file(s3_key, temp_path)
            if not success:
                raise ValueError(f"Failed to download Python file from S3: {s3_key}")
            module_name = os.path.splitext(os.path.basename(s3_key))[0]
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            os.unlink(temp_path)
            
            return module
        except Exception as e:
            print(f"Error loading Python module from S3: {str(e)}")
            raise
        
    def get_numpy_data(self, dataset_name, folder_type):
        if folder_type not in ['clean', 'adversarial']:
            raise ValueError(f"Invalid folder type for numpy data: {folder_type}")
            
        prefix = f"{dataset_name}/{folder_type}/"
        s3_files = self.list_images(prefix)
        
        # Filter only .npy files
        npy_files = [f for f in s3_files if f.endswith('.npy')]
        
        if not npy_files:
            print(f"Warning: No .npy files found in {prefix}")
            return {}
            
        numpy_data = {}
        
        for s3_key in npy_files:
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                file_content = response['Body'].read()
                
                array_data = np.load(io.BytesIO(file_content), allow_pickle=True)
                
                file_name = os.path.basename(s3_key)
                numpy_data[file_name] = array_data
                
            except (NoCredentialsError, ClientError) as e:
                print(f"Error getting numpy data from S3: {str(e)}")
                continue
                
        return numpy_data