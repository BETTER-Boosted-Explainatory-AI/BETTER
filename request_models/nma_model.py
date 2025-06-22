from pydantic import BaseModel
    
class NMAResult(BaseModel):
    message: str

class UploadURLResponse(BaseModel):
    upload_url: str
    key: str
    model_id: str

class InitiateMultipartUploadRequest(BaseModel):
    filename: str

class InitiateMultipartUploadResponse(BaseModel):
    upload_id: str
    key: str
    model_id: str

class PresignedPartUrlRequest(BaseModel):
    key: str
    upload_id: str
    part_number: int

class PresignedPartUrlResponse(BaseModel):
    url: str

class CompleteMultipartUploadRequest(BaseModel):
    key: str
    upload_id: str
    parts: list  # List of {"ETag": "...", "PartNumber": ...}
