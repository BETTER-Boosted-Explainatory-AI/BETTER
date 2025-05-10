from pydantic import BaseModel
from typing import List
    
class NMAResult(BaseModel):
    data: List[List[float]]
