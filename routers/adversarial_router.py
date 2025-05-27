from fastapi import APIRouter, HTTPException, status, Form, UploadFile, Depends
from services.adversarial_attacks_service import create_logistic_regression_detector, detect_adversarial_image, analysis_adversarial_image, does_detector_exist_
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from typing import List, Optional
from request_models.adversarial_model import DetectorResponse, AnalysisResult, DetectionResult
import logging
from utilss import debug_utils

# Set up logging
logger = logging.getLogger(__name__)

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
    current_user: User = Depends(require_authenticated_user)  
):
    try:
        logger.info("Starting to generate adversarial detector")

        # Validate clean_images
        if clean_images:
            for file in clean_images:
                if not file.filename.endswith(".npy"):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid file type for clean_images: {file.filename}. Only .npy files are allowed."
                    )

        # Validate adversarial_images
        if adversarial_images:
            for file in adversarial_images:
                if not file.filename.endswith(".npy"):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid file type for adversarial_images: {file.filename}. Only .npy files are allowed."
                    )
                
        detector = create_logistic_regression_detector(
            current_model_id, graph_type, clean_images, adversarial_images, current_user
        )

        if detector is None:
            raise HTTPException(status_code=404, detail="Detector was not created")
        
        # Return an instance of DetectorResponse
        return DetectorResponse(result="Detector created successfully")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@adversarial_router.post(
    "/adversarial/detect",
    status_code=status.HTTP_200_OK,
    response_model=DetectionResult,
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
    current_user: User = Depends(require_authenticated_user)
):
    try:
        image_content = await image.read()
        detection_result = detect_adversarial_image(current_model_id, graph_type, image_content, current_user)
        if detection_result is None:
            raise HTTPException(status_code=404, detail="Detection result not found")

        # Create a DetectionResult instance using the updated model definition
        final_result = DetectionResult(
            image=detection_result["image"],
            predictions=detection_result["predictions"],
            result=detection_result["result"]
        )

        return final_result.model_dump()
    except Exception as e:
        raise 
    
@adversarial_router.post(
    "/adversarial/analyze",
    status_code=status.HTTP_200_OK,
    response_model=AnalysisResult,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def analyze_adversarial(
    current_model_id: str = Form(...),
    graph_type: str = Form(...),
    image: UploadFile = Form(...),
    attack_type: str = Form(...),
    epsilon: Optional[float] = Form(None),
    alpha: Optional[float] = Form(None),
    overshoot: Optional[float] = Form(None),
    num_steps: Optional[int] = Form(None),
    classes_number: Optional[int] = Form(None),
    current_user: User = Depends(require_authenticated_user)
):
    try:
        image_content = await image.read()
        analysis_result = analysis_adversarial_image(
        model_id=current_model_id,
        graph_type=graph_type,
        attack_type=attack_type,
        image=image_content,
        user=current_user,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        overshoot=overshoot,
        num_classes=classes_number
        )
        if analysis_result is None:
            raise HTTPException(status_code=404, detail="Detection result not found")

        # # Convert original predictions to DetectionResult objects
        # original_predictions_result = [
        #     DetectionResult(label=k_label, probability=float(k_prob))
        #     for k_label, k_prob in analysis_result["original_predictions"]
        # ]

        # # Convert adversarial predictions to DetectionResult objects
        # adversarial_predictions_result = [
        #     DetectionResult(label=k_label, probability=float(k_prob))
        #     for k_label, k_prob in analysis_result["adversarial_predictions"]
        # ]

        # Create the AnalysisResult object
        # result = AnalysisResult(
        #     original_image=analysis_result["original_image"],
        #     original_predicition=original_predictions_result,
        #     original_verbal_explaination=analysis_result["original_verbal_explaination"],
        #     adversarial_image=analysis_result["adversarial_image"],
        #     adversarial_prediction=adversarial_predictions_result,
        #     adversarial_verbal_explaination=analysis_result["adversarial_verbal_explaination"],
        # )
        
        result = AnalysisResult(
            original_image=analysis_result["original_image"],
            original_predicition=analysis_result["original_predictions"],
            original_verbal_explaination=analysis_result["original_verbal_explaination"],
            adversarial_image=analysis_result["adversarial_image"],
            adversarial_prediction=analysis_result["adversarial_predictions"],
            adversarial_verbal_explaination=analysis_result["adversarial_verbal_explaination"],
        )

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@adversarial_router.get(
    "/adversarial/does_detector_exist",
    status_code=status.HTTP_200_OK,
    response_model=bool,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def does_detector_exist(
    current_model_id,
    graph_type,
    current_user: User = Depends(require_authenticated_user)
):
    try:
        logger.info("Checking if detector exists")
        detector_exists = does_detector_exist_(current_model_id, graph_type, current_user)
        logger.info(f"Detector exists: {detector_exists}")
        return detector_exists
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))