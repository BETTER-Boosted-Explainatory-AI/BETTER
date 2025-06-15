from pydantic import BaseModel
from typing import List, Optional, Tuple

class QueryResponse(BaseModel):
    query_result: Optional[List[str]]
    top_predictions: List[Tuple[str, float]] 
    image: str  # Base64-encoded string of the image
