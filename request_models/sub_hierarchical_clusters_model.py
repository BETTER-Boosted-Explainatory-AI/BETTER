from pydantic import BaseModel
from typing import List

class SubHierarchicalClusterRequest(BaseModel):
    dataset: str
    selected_labels: List[str]
    z_filename: str    