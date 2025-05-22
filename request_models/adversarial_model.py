from pydantic import BaseModel
from typing import List, Tuple

class DetectorResponse(BaseModel):
    result: str

# class PredictionResult(BaseModel):
#     label: str
#     probability: float

class AnalysisResult(BaseModel):
    original_image: str  # Base64-encoded string of the original image
    original_predicition: List[Tuple[str, float]] 
    original_verbal_explaination: List[str]
    adversarial_image: str  # Base64-encoded string of the adversarial image
    adversarial_prediction: List[Tuple[str, float]] 
    adversarial_verbal_explaination: List[str]

class DetectionResult(BaseModel):
    image: str
    predictions: List[Tuple[str, float]] 
    result: str


