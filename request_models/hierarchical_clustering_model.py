from pydantic import BaseModel
from typing import List

class HierarchicalClusterRequest(BaseModel):
    model_filename: str
    graph_type: str
    dataset: str
    
    
class HierarchicalClusterResult(BaseModel):
    data: List[List[float]]