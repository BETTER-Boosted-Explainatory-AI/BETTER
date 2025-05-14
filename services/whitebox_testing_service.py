from utilss.classes.whitebox_testing import WhiteBoxTesting
from utilss.classes.edges_dataframe import EdgesDataframe
from .models_service import _get_model_path, _get_model_filename, get_user_models_info
from fastapi import HTTPException, status
import os

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
    model_info = get_user_models_info(current_user.get_models_json_path(), current_model_id)

    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    dataset_str = model_info["dataset"]
    
    edges_df_filename = _get_edges_dataframe_path(current_user.get_user_id(), current_model_id, graph_type)
    model_filename = _get_model_filename(user_id, current_model_id, graph_type)
    
    edges_data = EdgesDataframe(model_filename, edges_df_filename)
    edges_data.load_dataframe()
    df = edges_data.get_dataframe()

    whitebox_testing = WhiteBoxTesting(model_filename)
    problematic_imgs = whitebox_testing.find_problematic_images(source_labels, target_labels, df, dataset_str)
    
    return problematic_imgs
