from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionResponse(BaseModel):
    prediction: Dict[str, Any] = Field(
        ..., 
        description="Prediction results from the model",
        example={"class": "dog", "confidence": 0.95}
    )