from fastapi import APIRouter, HTTPException, status
from request_models.pgd_attack_models import PGDAttackRequest, PGDAttackResponse
from services.pgd_attack_service import perform_pgd_attack
from enums.hierarchical_cluster_types import HierarchicalClusterType

# Create router for PGD attack endpoints
pgd_attack_router = APIRouter()

@pgd_attack_router.post(
    "/attack/pgd",
    response_model=PGDAttackResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Bad request"},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def run_pgd_attack(attack_request: PGDAttackRequest) -> PGDAttackResponse:
    """
    Run a Projected Gradient Descent (PGD) attack on a specified image.
    
    The endpoint accepts parameters to customize the attack and returns detailed results
    including attack success, adversarial scores, and perturbation metrics.
    """
    try:
        # Run the PGD attack using the service
        attack_results = perform_pgd_attack(
            dataset_name=attack_request.dataset_name,
            image_index=attack_request.image_index,
            epsilon=attack_request.epsilon,
            alpha=attack_request.alpha,
            num_steps=attack_request.num_steps,
            targeted=attack_request.targeted,
            target_class=attack_request.target_class,
            threshold=attack_request.threshold,
            cluster_type=HierarchicalClusterType.SIMILARITY  # Default to similarity-based clustering
        )
        
        # Return the results
        return PGDAttackResponse(data=attack_results)
    
    except ValueError as e:
        # Handle validation errors (e.g., invalid dataset name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except FileNotFoundError as e:
        # Handle resource not found errors
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )


@pgd_attack_router.get(
    "/attack/pgd/types",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
async def get_cluster_types():
    """
    Get available hierarchical clustering types for PGD attack.
    """
    try:
        # Return all available clustering types
        cluster_types = {
            "SIMILARITY": HierarchicalClusterType.SIMILARITY.value,
            "DISSIMILARITY": HierarchicalClusterType.DISSIMILARITY.value,
            "CONFUSION_MATRIX": HierarchicalClusterType.CONFUSION_MATRIX.value
        }
        
        return {"status": "success", "data": cluster_types}
    
    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )