from pydantic import BaseModel

class DetectorResponse(BaseModel):
    result: str

class DetectionResult(BaseModel):
    label: str
    probability: float
