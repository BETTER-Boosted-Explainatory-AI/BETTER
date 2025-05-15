from fastapi import HTTPException, status
from PIL import Image
import io
import numpy as np
import os
from utilss.classes.whitebox_testing import WhiteBoxTesting
from utilss.classes.edges_dataframe import EdgesDataframe
from .models_service import _get_model_path, _get_model_filename, get_user_models_info
from .dataset_service import _get_dataset_config, _load_dataset
from utilss.photos_utils import encode_image_to_base64


def _get_edges_dataframe_path(user_id, model_id, graph_type):
    model_path = _get_model_path(user_id, model_id)
    edges_filename = f'edges_df.csv'
    edges_df_path = os.path.join(model_path, graph_type, edges_filename)
    if not os.path.exists(edges_df_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not find edges dataframe file"
        )

    return edges_df_path


def get_white_box_analysis(current_user, current_model_id, graph_type, source_labels, target_labels):
    user_id = current_user.get_user_id()
    model_info = get_user_models_info(current_user, current_model_id)

    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    dataset_str = model_info["dataset"]

    dataset_config = _get_dataset_config(dataset_str)
    dataset = _load_dataset(dataset_config)

    edges_df_filename = _get_edges_dataframe_path(
        user_id, current_model_id, graph_type)
    model_filename = _get_model_filename(user_id, current_model_id, graph_type)

    edges_data = EdgesDataframe(model_filename, edges_df_filename)
    edges_data.load_dataframe()
    df = edges_data.get_dataframe()

    whitebox_testing = WhiteBoxTesting(model_filename)
    problematic_imgs_dict = whitebox_testing.find_problematic_images(
        source_labels, target_labels, df, dataset_str)
    imgs_list = []

    # for image_id, matches in problematic_imgs_dict.items():
    #     img, label = dataset.get_train_image_by_id(image_id)
    #     pil_image = Image.open(io.BytesIO(img)).convert("RGB")
    #     image_array = np.array(pil_image)
    #     original_image_base64 = encode_image_to_base64(image_array)
        
    #     imgs_list.append({
    #         "image": original_image_base64,
    #         "image_id": image_id,
    #         "matches": matches,
    #     })
    
    for image_id, matches in problematic_imgs_dict.items():
        img, label = dataset.get_train_image_by_id(image_id)
        
        # If img is a numpy array (which is likely the case)
        if isinstance(img, np.ndarray):
            # Convert numpy array directly to PIL Image
            pil_image = Image.fromarray(img)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
        else:
            # Only try BytesIO if you're sure img contains binary image data
            try:
                pil_image = Image.open(io.BytesIO(img)).convert("RGB")
            except Exception:
                # Handle the error or provide a fallback
                continue
                
        image_array = np.array(pil_image)
        original_image_base64 = encode_image_to_base64(image_array)
        
        imgs_list.append({
            "image": original_image_base64,
            "image_id": image_id,
            "matches": matches,
        })

    return imgs_list
