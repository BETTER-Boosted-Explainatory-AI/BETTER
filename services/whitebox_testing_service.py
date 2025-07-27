from fastapi import HTTPException, status
import numpy as np
import os
from utilss.classes.whitebox_testing import WhiteBoxTesting
from utilss.classes.edges_dataframe import EdgesDataframe
from .models_service import _get_model_path, _get_model_filename, get_user_models_info
from .dataset_service import _load_dataset
from utilss.photos_utils import encode_image_to_base64
from utilss.s3_utils import get_users_s3_client 

def _get_edges_dataframe_path(user_id, model_id, graph_type):
    # Initialize S3 client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    # Get the model path (which should now be an S3 key)
    model_path = _get_model_path(user_id, model_id)
    
    # Construct the S3 key for the edges dataframe
    edges_filename = 'edges_df.csv'
    edges_df_path = f'{model_path}/{graph_type}/{edges_filename}'
    

    try:
        s3_client.head_object(Bucket=s3_bucket, Key=edges_df_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not find edges dataframe file in S3: {str(e)}"
        )
    
    return edges_df_path

def get_white_box_analysis(current_user, current_model_id, graph_type, source_labels, target_labels):
    # Initialize S3 client
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    if current_model_id is None or graph_type is None or source_labels is None or target_labels is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameters"
        )
    
    user_id = current_user.get_user_id()
    model_info = get_user_models_info(current_user, current_model_id)
    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    dataset_str = model_info["dataset"]
    dataset = _load_dataset(dataset_str)

    # Get file paths (now S3 keys)
    edges_df_filename = _get_edges_dataframe_path(
        user_id, current_model_id, graph_type)
    model_filename = _get_model_filename(user_id, current_model_id, graph_type)

    # Create objects with S3 support
    edges_data = EdgesDataframe(model_filename, edges_df_filename)
    edges_data.load_dataframe()
    df = edges_data.get_dataframe()

    whitebox_testing = WhiteBoxTesting(model_filename)
    problematic_imgs_dict = whitebox_testing.find_problematic_images(
        source_labels, target_labels, df, dataset_str)
    imgs_list = []
    
    for image_id, matches in problematic_imgs_dict.items():
        try:
            img, _ = dataset.get_train_image_by_id(image_id)
            if dataset_str.lower() == "imagenet":
                # Fetch the full image key and extract only the filename
                full_image_key = dataset.get_image_key_by_id(image_id)
                image_filename = os.path.basename(full_image_key)  # Extract the filename
            else:
                image_filename = image_id
        
        
            # Process the image for encoding
            if img.max() <= 1.0:
                # If normalized to [0,1], scale to [0,255]
                img = (img * 255).astype(np.uint8)
            
            # Directly encode the properly processed image
            original_image_base64 = encode_image_to_base64(img)
            
            imgs_list.append({
                "image": original_image_base64,
                "image_id": image_filename,
                "matches": matches,
            })
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            # Optionally continue to next image or handle the error
    
    return imgs_list
