from pydantic import BaseModel
from typing import List

class DetectorResponse(BaseModel):
    result: str

class DetectionResult(BaseModel):
    label: str
    probability: float

class AnalysisResult(BaseModel):
    original_image: str  # Base64-encoded string of the original image
    original_predicition: List[DetectionResult]
    adversarial_image: str  # Base64-encoded string of the adversarial image
    adversarial_prediction: List[DetectionResult]
