from pydantic import BaseModel
from typing import List
    
class NMAResult(BaseModel):
    message: str