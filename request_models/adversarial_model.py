from pydantic import BaseModel

class DetectorResponse(BaseModel):
    result: str

class DetectionResult(BaseModel):
    label: str
    probability: float

class AnalysisResult(BaseModel):
    original_image: str  # Base64-encoded string of the original image
    adversarial_image: str  # Base64-encoded string of the adversarial image
