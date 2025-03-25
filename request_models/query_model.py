from pydantic import BaseModel
from typing import List, Optional

class QueryResponse(BaseModel):
    query_result: Optional[List[str]]