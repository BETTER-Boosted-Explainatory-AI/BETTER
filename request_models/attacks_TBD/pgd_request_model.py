from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PGDAttackRequest(BaseModel):
    """Request model for PGD attack."""
    dataset_name: str = Field(..., description="Name of the dataset (e.g., 'cifar100', 'imagenet')")
    image_index: int = Field(..., description="Index of the image to attack")
    epsilon: float = Field(0.1, description="Maximum perturbation size (default: 0.1)")
    alpha: float = Field(0.01, description="Step size for each iteration (default: 0.01)")
    num_steps: int = Field(40, description="Number of attack iterations (default: 40)")
    targeted: bool = Field(False, description="Whether to perform a targeted attack")
    target_class: Optional[int] = Field(None, description="Target class for targeted attacks (if targeted=True)")
    threshold: Optional[float] = Field(None, description="Custom threshold for adversarial detection")

class PGDAttackResult(BaseModel):
    """Response model for PGD attack results."""
    attack_type: str
    original_class: int
    original_class_name: str
    adversarial_class: int
    adversarial_class_name: str
    attack_success: bool
    original_score: float
    original_is_adversarial: bool
    adversarial_score: float
    adversarial_is_adversarial: bool
    detection_success: bool
    l2_norm: float
    linf_norm: float
    execution_time: float
    parameters: Dict[str, Any]

class PGDAttackResponse(BaseModel):
    """Wrapper response model with status and data."""
    status: str = "success"
    data: PGDAttackResult