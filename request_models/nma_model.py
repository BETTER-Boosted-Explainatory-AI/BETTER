from pydantic import BaseModel
from typing import List

class NMARequest(BaseModel):
    model_filename: str
    graph_type: str
    dataset: str
    
    
class NMAResult(BaseModel):
    data: List[List[float]]
