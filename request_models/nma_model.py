from pydantic import BaseModel
    
class NMAResult(BaseModel):
    message: str

class UploadURLRequest(BaseModel):
    filename: str

class UploadURLResponse(BaseModel):
    upload_url: str
    key: str
    model_id: str