from fastapi import APIRouter, HTTPException, status, Form, UploadFile, Depends
from services.adversarial_attacks_service import create_logistic_regression_detector, detect_adversarial_image, analysis_adversarial_image, does_detector_exist_, get_detector_list
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from typing import List, Optional
from request_models.adversarial_model import DetectorResponse, AnalysisResult, DetectionResult, DetectorListRequest
import logging

# Set up logging
logger = logging.getLogger(__name__)

adversarial_router = APIRouter()

@adversarial_router.post(
    "/api/adversarial/generate",
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
                if not file.filename or not file.filename.endswith(".npy"):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid file type for clean_images: {file.filename}. Only .npy files are allowed."
                    )

        # Validate adversarial_images
        if adversarial_images:
            for file in adversarial_images:
                if not file.filename or not file.filename.endswith(".npy"):
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
    "/api/adversarial/detect",
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
    detector_filename: str = Form(...),
    current_user: User = Depends(require_authenticated_user)
):
    try:
        image_content = await image.read()
        detection_result = detect_adversarial_image(current_model_id, graph_type, image_content, current_user, detector_filename)
        if detection_result is None:
            raise HTTPException(status_code=404, detail="Detection result not found")

        final_result = DetectionResult(
            image=detection_result["image"],
            predictions=detection_result["predictions"],
            result=detection_result["result"],
            probability=detection_result["probability"]
        )

        return final_result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@adversarial_router.post(
    "/api/adversarial/analyze",
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
    detector_filename: str = Form(...),
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
        num_classes=classes_number,
        detector_filename=detector_filename
        )
        if analysis_result is None:
            raise HTTPException(status_code=404, detail="Detection result not found")

        # Create the AnalysisResult object
        result = AnalysisResult(
            original_image=analysis_result["original_image"],
            original_predicition=analysis_result["original_predictions"],
            original_verbal_explaination=analysis_result["original_verbal_explaination"],
            original_probability=analysis_result["original_probability"],
            original_detection_result=analysis_result["original_detection_result"],
            adversarial_image=analysis_result["adversarial_image"],
            adversarial_prediction=analysis_result["adversarial_predictions"],
            adversarial_verbal_explaination=analysis_result["adversarial_verbal_explaination"],
            adversarial_probability=analysis_result["adversarial_probability"],
            adversarial_detection_result=analysis_result["adversarial_detection_result"]
        )

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@adversarial_router.get(
    "/api/adversarial/does_detector_exist",
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


@adversarial_router.post(
    "/api/adversarial/get_detectors",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_detectors(
    detector_list_request: DetectorListRequest,
    current_user: User = Depends(require_authenticated_user)
):
    try:
        logger.info("Fetching adversarial detectors")
        current_model_id = detector_list_request.current_model_id
        graph_type = detector_list_request.graph_type
        detectors_list = get_detector_list(current_user, current_model_id, graph_type)
        
        if not detectors_list:
            raise HTTPException(status_code=404, detail="No detectors found")
        
        return detectors_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    