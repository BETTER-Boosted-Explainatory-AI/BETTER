import numpy as np
from fastapi import HTTPException, status
from .models_service import _check_model_path
from utilss.classes.dendrogram import Dendrogram
import json
# from data.datasets.cifar100_info import CIFAR100_INFO
# from data.datasets.imagenet_info import IMAGENET_INFO
from services.dataset_service import _get_dataset_config
from .models_service import get_user_models_info
import boto3
import os
from utilss.s3_utils import get_users_s3_client 

def _get_dendrogram_path(user_id, model_id, graph_type):
    model_path = _check_model_path(user_id, model_id, graph_type)
    dendrogram_filename = f'{model_path}/{graph_type}/dendrogram'
    return dendrogram_filename


### original implemetation ###
# def _get_sub_dendrogram(current_user, model_id, graph_type, selected_labels):
#     dendrogram_filename = _get_dendrogram_path(current_user.user_id, model_id, graph_type)
    
#     if selected_labels is None or selected_labels == []:   
#         model_info = get_user_models_info(current_user, model_id)
#         dataset = model_info["dataset"]
        
#         # if dataset == "cifar100":
#         #     selected_labels = CIFAR100_INFO["init_selected_labels"]
#         # elif dataset == "imagenet":
#         #     selected_labels = IMAGENET_INFO["init_selected_labels"]
        
#         if dataset == "cifar100":
#             selected_labels = _get_dataset_config(dataset)["init_selected_labels"]
#         elif dataset == "imagenet":
#             selected_labels = _get_dataset_config(dataset)["init_selected_labels"]
#         else:
#             raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Dataset not supported")
        
#     dendrogram = Dendrogram(dendrogram_filename)
#     dendrogram.load_dendrogram()
#     sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
#     sub_dendrogram_json = json.loads(sub_dendrogram)
#     return sub_dendrogram_json, selected_labels



### S3 implementation ### 
def _get_sub_dendrogram(current_user, model_id, graph_type, selected_labels):
    # Get the S3 path to the dendrogram
    dendrogram_filename = _get_dendrogram_path(current_user.user_id, model_id, graph_type)
    
    if selected_labels is None or selected_labels == []:   
        model_info = get_user_models_info(current_user, model_id)
        dataset = model_info["dataset"]
        
        if dataset == "cifar100":
            selected_labels = _get_dataset_config(dataset)["init_selected_labels"]
        elif dataset == "imagenet":
            selected_labels = _get_dataset_config(dataset)["init_selected_labels"]
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Dataset not supported")
    
    # Initialize S3 client
    s3_client = get_users_s3_client() 
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    # Set up S3 params for Dendrogram class
    s3_params = {
        "s3_client": s3_client,
        "s3_bucket": s3_bucket
    }
    
    # Create and load dendrogram with S3 support
    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json, selected_labels

### original implemetation ###
# def _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name):
#     dendrogram_filename = _get_dendrogram_path(user_id, model_id, graph_type)
    
#     dendrogram = Dendrogram(dendrogram_filename)
#     dendrogram.load_dendrogram()
#     dendrogram.rename_cluster(cluster_id, new_name)
#     dendrogram.save_dendrogram()
#     sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
#     sub_dendrogram_json = json.loads(sub_dendrogram)
#     return sub_dendrogram_json
    
  
### S3 implementation ###  
def _rename_cluster(user_id, model_id, graph_type, selected_labels, cluster_id, new_name):
    # Get the S3 path to the dendrogram
    dendrogram_filename = _get_dendrogram_path(user_id, model_id, graph_type)
    
    # Initialize S3 client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    # Set up S3 params for Dendrogram class
    s3_params = {
        "s3_client": s3_client,
        "s3_bucket": s3_bucket
    }
    
    # Create and load dendrogram with S3 support
    dendrogram = Dendrogram(dendrogram_filename)
    dendrogram.load_dendrogram()
    dendrogram.rename_cluster(cluster_id, new_name)
    dendrogram.save_dendrogram()
    sub_dendrogram = dendrogram.get_sub_dendrogram_formatted(selected_labels)
    sub_dendrogram_json = json.loads(sub_dendrogram)
    return sub_dendrogram_json