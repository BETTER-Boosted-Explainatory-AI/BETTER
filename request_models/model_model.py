from pydantic import BaseModel
from typing import List, Optional

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

class CurrentModelRequest(BaseModel):
    model_id: str
    graph_type: str