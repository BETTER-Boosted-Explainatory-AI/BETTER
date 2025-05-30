from pydantic import BaseModel
from typing import List

class DatasetLabelsResult(BaseModel):
    data: List[str]
