import os
import json
import uuid
import boto3
import io
from utilss.s3_utils import get_users_s3_client 

class User:
    def __init__(self, user_id: uuid, email: str, models: list = None):
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.email = email
        self.models = models if models is not None else []
        self.current_model = None
        
        # S3 client setup
        self.s3_client = get_users_s3_client()
        self.s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
        if not self.s3_bucket:
            raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
        
        self.users_json_path = "users.json"  # At bucket root
        self.user_folder_path = f"{self.user_id}"  # Just the user ID
        self.models_json_path = f"{self.user_id}/models.json"
        self.current_model_json = f"{self.user_id}/current_model.json"

    def create_user(self):
        # Create empty models.json
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=self.models_json_path,
            Body=json.dumps([], indent=4)
        )
        
        # Create empty current_model.json
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=self.current_model_json,
            Body=json.dumps({}, indent=4)
        )

    def load_models(self):
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.models_json_path)
            self.models = json.loads(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            print(f"No models found for user {self.user_id}, creating empty models file")
            # Create the user's models file if it doesn't exist
            self.models = []
            self.create_user()
        except Exception as e:
            print(f"Error loading models for user {self.user_id}: {str(e)}")
            self.models = []

    def get_user_id(self):
        return self.user_id
    
    def get_models(self):
        return self.models
    
    def get_models_json_path(self):
        return self.models_json_path

    def add_model(self, model_info: dict):
        self.models.append(model_info)
        
        # Write updated models list back to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=self.models_json_path,
            Body=json.dumps(self.models, indent=4)
        )
            
    def set_current_model(self, model_info: dict):
        self.current_model = model_info
        
        # Write current model to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=self.current_model_json,
            Body=json.dumps(self.current_model, indent=4)
        )
        
        return self.current_model
    
    def load_current_model(self):
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.current_model_json)
            self.current_model = json.loads(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            print(f"No current model found for user {self.user_id}")
            self.current_model = {}
        except Exception as e:
            print(f"Error loading current model for user {self.user_id}: {str(e)}")
            self.current_model = {}

    def get_current_model(self):
        if self.current_model is None:
            self.load_current_model()
        return self.current_model
    
    def get_user_folder(self):
        return self.user_folder_path
  
    def find_user_in_db(self):
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.users_json_path)
            users = json.loads(response['Body'].read().decode('utf-8'))
            
            for user in users:
                if user["id"] == self.user_id:
                    return User(user_id=user["id"], email=user["email"])
            
            return None
        except self.s3_client.exceptions.NoSuchKey:
            return None
