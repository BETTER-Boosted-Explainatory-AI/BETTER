from pydantic import BaseModel
from typing import List, Optional

class SubHierarchicalClusterRequest(BaseModel):
    dataset: str
    selected_labels: List[str]
    z_filename: str    

class ClusterNode(BaseModel):
    id: int
    name: str
    value: Optional[float] = None
    children: Optional[List['ClusterNode']] = None

    class Config:
        orm_mode = True

class SubHierarchicalClusterResult(BaseModel):
    data: ClusterNode