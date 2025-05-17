from pydantic import BaseModel
from typing import List, Optional

class ModelRequest(BaseModel):
    model_id: str
    file_name: str
    dataset: str
    graph_type: str
    
class ModelResult(BaseModel):
    model_id: str
    file_name: str
    dataset: str
    graph_type: List[str]
