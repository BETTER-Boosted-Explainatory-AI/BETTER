from fastapi import APIRouter, HTTPException, status
from request_models.dataset_model import DatasetLabelsResult

from services.dataset_service import _get_dataset_labels

datasets_router = APIRouter()

@datasets_router.get(
    "/datasets/{dataset_name}/labels", 
    response_model=DatasetLabelsResult,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_dataset_labels(dataset_name: str) -> DatasetLabelsResult:
    dataset_labels = _get_dataset_labels(dataset_name)  # Pass dataset_name correctly

    if dataset_labels is None:
        raise HTTPException(status_code=404, detail="Couldn't find dataset labels")
    
    return DatasetLabelsResult(data=dataset_labels)
