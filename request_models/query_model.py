from pydantic import BaseModel
from typing import List, Optional, Tuple

class QueryResponse(BaseModel):
    query_result: Optional[List[str]]
    top_3_predictions: List[Tuple[str, float]] 