import numpy as np
from fastapi import HTTPException, Request, status
from .dataset_service import _get_dataset_config, _load_dataset


def _get_sub_heirarchical_clustering(dataset_str, selected_labels, z_filename):
    if z_filename is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Model filename is required"
        )
    
    dataset_config = _get_dataset_config(dataset_str)