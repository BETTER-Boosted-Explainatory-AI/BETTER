from fastapi import APIRouter, Depends, HTTPException, status, Depends
from services.whitebox_testing_service import get_white_box_analysis
from utilss.classes.user import User
from services.users_service import require_authenticated_user
from request_models.whitebox_testing_model import WhiteboxTestingRequest

whitebox_testing_router = APIRouter()


@whitebox_testing_router.post(
    "/whitebox_testing",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
    }
)
async def get_whitebox_testing(request: WhiteboxTestingRequest,
                               current_user: User = Depends(
                                   require_authenticated_user)
                               ):

    problematic_imgs = get_white_box_analysis(
        current_user,
        request.model_id,
        request.graph_type,
        request.source_labels,
        request.target_labels)

    if problematic_imgs is None:
        raise HTTPException(
            status_code=404, detail="White Box Testing was not created")

    return problematic_imgs
