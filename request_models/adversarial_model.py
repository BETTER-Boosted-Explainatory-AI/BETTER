from pydantic import BaseModel

class DetectorResponse(BaseModel):
    message: str