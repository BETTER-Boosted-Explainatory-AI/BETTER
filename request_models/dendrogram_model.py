from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import uuid


class DendrogramRequest(BaseModel):
    model_id: uuid.UUID
    graph_type: str
    selected_labels: Optional[List[str]] = None

class DendrogramResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    value: Optional[float] = None
    children: Optional[List['DendrogramResult']] = None
    selected_labels: Optional[List[str]] = None

class NamingClusterRequest(BaseModel):
    model_id: uuid.UUID
    graph_type: str
    selected_labels: List[str]
    cluster_id: int
    new_name: str
