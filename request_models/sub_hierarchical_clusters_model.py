from pydantic import BaseModel
from typing import List, Optional

class SubHierarchicalClusterRequest(BaseModel):
    dataset: str
    selected_labels: List[str]
    z_filename: str    

class SubHierarchicalClusterResult(BaseModel):
    id: int
    name: str
    value: Optional[float] = None
    children: Optional[List['SubHierarchicalClusterResult']] = None

    class Config:
        from_attributes = True

class NamingClusterRequest(BaseModel):
    cluster_id: int
    dendrogram_filename: str
    new_name: str

class NamingClusterResult(BaseModel):
    message: str
