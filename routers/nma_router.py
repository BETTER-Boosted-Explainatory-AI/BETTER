from fastapi import APIRouter, HTTPException, status, Form, Depends
from request_models.nma_model import NMAResult, InitiateMultipartUploadRequest, InitiateMultipartUploadResponse, PresignedPartUrlRequest, PresignedPartUrlResponse, CompleteMultipartUploadRequest
from utilss.files_utils import _update_model_metadata
from services.users_service import require_authenticated_user
from utilss.classes.user import User
from utilss.enums.graph_types import GraphTypes
from utilss.aws_job_utils import submit_nma_batch_job
from services.models_service import initiate_multipart_upload, presigned_part_url, finalize_multipart_upload, ensure_model_ready_in_s3, get_user_models_info


nma_router = APIRouter()

def _validate_graph_type(graph_type: str):
    if graph_type not in [
        GraphTypes.SIMILARITY.value,
        GraphTypes.DISSIMILARITY.value,
        GraphTypes.COUNT.value,
    ]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Graph type must be either 'similarity', 'dissimilarity', or 'count'"
        )
        
def _handle_nma_submission(
    current_user: User,
    dataset: str,
    graph_type: str,
    min_confidence: float,
    top_k: int,
    model_id: str = None,
    model_filename: str = None
) -> NMAResult:
    _validate_graph_type(graph_type)
    if model_filename is None and model_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model file or model_id is required"
        )
        
    model_file = get_user_models_info(current_user, model_id)    
    model_filename_f = model_file.get("model_filename") if model_file else None
    if(model_filename_f is not None):
        model_filename = model_filename_f
    model_id_f = model_id 

    key = f"{current_user.user_id}/{model_id_f}/{model_filename}"
    print(f"key: {key}")

    # if(ensure_model_ready_in_s3(key)):
    job_id = submit_nma_batch_job(current_user.user_id, model_id_f, dataset, graph_type, min_confidence, top_k)        
    
    print(f"Submitting NMA job with parameters: {current_user.user_id}, {model_filename},{graph_type}")
    
    print(f"Submitted NMA job with ID: {job_id}")
    metadata_result = _update_model_metadata(
        current_user, model_id_f, model_filename, dataset, graph_type, min_confidence, top_k, job_id)
    if not metadata_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model metadata"
        )
    message = {"message": "NMA job has been submitted successfully."}
    return NMAResult(**message)

@nma_router.post(
    "/api/nma/{model_id}",
    response_model=NMAResult,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def create_nma_by_id(
    model_id: str,
    current_user: User = Depends(require_authenticated_user),
    dataset: str = Form(...),
    graph_type: str = Form(...),
    min_confidence: float = Form(0.5),
    top_k: int = Form(5),
    model_filename: str = Form(...)
) -> NMAResult:
    try:
        return _handle_nma_submission(
            current_user, dataset, graph_type, min_confidence, top_k, model_id, model_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@nma_router.post(
    "/api/initiate-multipart-upload",
    response_model=InitiateMultipartUploadResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    }
)

def start_multipart_upload(
    body: InitiateMultipartUploadRequest,
    current_user: User = Depends(require_authenticated_user)
):
    """
    Initiate a multipart upload for a model file.
    """
    try:
        model_name = body.filename
        result = initiate_multipart_upload(
            current_user, model_name)
        return InitiateMultipartUploadResponse(upload_id=result["upload_id"], model_id=result["model_id"], key=result["key"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@nma_router.post(
    "/api/presigned-part-url",
    response_model=PresignedPartUrlResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    }
)
def get_presigned_part_url(
    body: PresignedPartUrlRequest,
    current_user: User = Depends(require_authenticated_user)
):
    """
    Get a presigned URL for a specific part of a multipart upload.
    """
    try:
        url = presigned_part_url(
            key=body.key,
            upload_id=body.upload_id,
            part_number=body.part_number
        )

        return PresignedPartUrlResponse(url=url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@nma_router.post(
    "/api/complete-multipart-upload",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    }
)
def complete_multipart_upload(
    body: CompleteMultipartUploadRequest,
    current_user: User = Depends(require_authenticated_user)
):
    """
    Complete a multipart upload for a model file.
    """
    try:
        print(f"Completing multipart upload with parts: {body.parts}")
        result = finalize_multipart_upload(
            key=body.key,
            upload_id=body.upload_id,
            parts=body.parts
        )
        return {"status": "completed", "location": result.get("Location")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))