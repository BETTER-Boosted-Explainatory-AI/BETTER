from pydantic import BaseModel
from typing import List, Tuple

class DetectorResponse(BaseModel):
    result: str

class AnalysisResult(BaseModel):
    original_image: str  # Base64-encoded string of the original image
    original_predicition: List[Tuple[str, float]] 
    original_verbal_explaination: List[str]
    original_probability: float
    original_detection_result: str  # Result of the original image detection
    adversarial_image: str  # Base64-encoded string of the adversarial image
    adversarial_prediction: List[Tuple[str, float]] 
    adversarial_verbal_explaination: List[str]
    adversarial_probability: float
    adversarial_detection_result: str  # Result of the adversarial image detection

class DetectionResult(BaseModel):
    image: str
    predictions: List[Tuple[str, float]] 
    result: str
    probability: float

class DetectorListRequest(BaseModel):
    current_model_id: str
    graph_type: str