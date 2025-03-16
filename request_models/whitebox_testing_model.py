from pydantic import BaseModel
from typing import List

class WhiteboxTestingRequest(BaseModel):
    model_filename: str
    source_labels: List[str]
    target_labels: List[str]
    edges_data_filename: str

