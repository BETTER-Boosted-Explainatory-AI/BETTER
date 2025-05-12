from fastapi import APIRouter, HTTPException, status, Form, UploadFile, Depends
from services.adversarial_attacks_service import create_logistic_regression_detector, detect_adversarial_image, analysis_adversarial_image
from services.users_service import get_current_session_user
from utilss.classes.user import User
from typing import List, Optional
from request_models.adversarial_model import DetectorResponse
import os

adversarial_router = APIRouter()

@adversarial_router.post(
    "/adversarial/generate",
    status_code=status.HTTP_201_CREATED,
    response_model=DetectorResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def generate_adversarial_detector(
    current_model_id: str = Form(...),
    graph_type: str = Form(...),
    clean_images: Optional[List[UploadFile]] = None, 
    adversarial_images: Optional[List[UploadFile]] = None,
    current_user: User = Depends(get_current_session_user)  
):
    try:
        BASE_DIR = os.getenv("USERS_PATH", "users")  # Base directory for user data
        user_folder = os.path.join(BASE_DIR, str(current_user.user_id))
        detector = create_logistic_regression_detector(
            current_model_id, graph_type, clean_images, adversarial_images, user_folder
        )

        if detector is None:
            raise HTTPException(status_code=404, detail="Detector was not created")
        
        # Return an instance of DetectorResponse
        return DetectorResponse(result="Detector created successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@adversarial_router.post(
    "/adversarial/detect",
    status_code=status.HTTP_200_OK,
    response_model=DetectorResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def detect_query(
    current_model_id: str = Form(...),
    graph_type: str = Form(...),
    image: UploadFile = Form(...),
    current_user: User = Depends(get_current_session_user)
):
    try:
        BASE_DIR = os.getenv("USERS_PATH", "users")
        user_folder = os.path.join(BASE_DIR, str(current_user.user_id))
        image_content = await image.read()
        detection_result = detect_adversarial_image(current_model_id, graph_type, image_content, user_folder)
        if detection_result is None:
            raise HTTPException(status_code=404, detail="Detection result not found")
        return DetectorResponse(result=detection_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @adversarial_router.post(
#     "/adversarial/analyze",
#     status_code=status.HTTP_200_OK,
#     response_model=DetectorResponse,
#     responses={
#         status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
#         status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
#         status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
#     }
# )
# async def analyze_adversarial(
#     current_model_id: str = Form(...),
#     graph_type: str = Form(...),
#     image: UploadFile = Form(...),
#     attack_type: str = Form(...),
#     epsilon: Optional[float] = Form(None),
#     alpha: Optional[float] = Form(None),
    
#     current_user: User = Depends(get_current_session_user)
# ):
#     try:
#         BASE_DIR = os.getenv("USERS_PATH", "users")
#         user_folder = os.path.join(BASE_DIR, str(current_user.user_id))
#         image_content = await image.read()
#         analysis_result = analysis_adversarial_image(current_model_id, graph_type, attack_type ,image_content, user_folder)
#         if analysis_result is None:
#             raise HTTPException(status_code=404, detail="Detection result not found")
#         return DetectorResponse(result=analysis_result)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)
# )
        
        

