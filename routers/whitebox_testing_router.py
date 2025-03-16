from fastapi import APIRouter, Depends, HTTPException, Request, status
from services.whitebox_testing_service import get_white_box_analysis
from request_models.whitebox_testing_model import WhiteboxTestingRequest

whitebox_testing_router = APIRouter()

@whitebox_testing_router.post(
    "/whitebox_testing", 
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_whitebox_testing(whitebox_testing_request: WhiteboxTestingRequest):
    model_name = whitebox_testing_request.model_filename
    source_labels = whitebox_testing_request.source_labels
    target_labels = whitebox_testing_request.target_labels
    edges_data = whitebox_testing_request.edges_data_filename
    problematic_imgs = get_white_box_analysis(model_name, source_labels, target_labels, edges_data)
    if problematic_imgs is None:
        raise HTTPException(status_code=404, detail="White Box Testing was not created")
    return problematic_imgs