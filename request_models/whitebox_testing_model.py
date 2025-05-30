from pydantic import BaseModel
from typing import List
import uuid

class WhiteboxTestingRequest(BaseModel):
    model_id: uuid.UUID
    graph_type: str
    source_labels: List[str]
    target_labels: List[str]
