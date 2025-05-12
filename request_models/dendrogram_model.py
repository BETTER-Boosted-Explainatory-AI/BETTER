from pydantic import BaseModel
from typing import List, Optional
import uuid


class DendrogramRequest(BaseModel):
    user_id: uuid.UUID
    model_id: uuid.UUID
    graph_type: str
    selected_labels: List[str]

class DendrogramResult(BaseModel):
    id: int
    name: str
    value: Optional[float] = None
    children: Optional[List['DendrogramResult']] = None

    class Config:
        from_attributes = True

class NamingClusterRequest(BaseModel):
    cluster_id: int
    dendrogram_filename: str
    new_name: str

class NamingClusterResult(BaseModel):
    message: str
