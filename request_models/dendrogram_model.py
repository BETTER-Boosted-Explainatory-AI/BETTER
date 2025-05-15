from pydantic import BaseModel
from typing import List, Optional
import uuid


class DendrogramRequest(BaseModel):
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
    model_id: uuid.UUID
    graph_type: str
    selected_labels: List[str]
    cluster_id: int
    new_name: str

