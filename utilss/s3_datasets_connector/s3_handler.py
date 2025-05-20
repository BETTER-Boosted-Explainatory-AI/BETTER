import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError


class S3Handler:
    """Handler for S3 operations."""
    
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
            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except (NoCredentialsError, ClientError) as e:
            print(f"Error downloading file from S3: {str(e)}")
            return False
    
    def list_objects(self, prefix=''):
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