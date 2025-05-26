from pydantic import BaseModel
from typing import List, Optional

class BatchJob(BaseModel):
    job_graph_type: str
    job_status: str

class ModelRequest(BaseModel):
    model_id: str
    file_name: str
    dataset: str
    graph_type: str

    
class ModelsResult(BaseModel):
    model_id: str
    file_name: str
    dataset: str
    graph_type: List[str]
    batch_jobs: Optional[List[BatchJob]]

class CurrentModelRequest(BaseModel):
    model_id: str
    graph_type: str