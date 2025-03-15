from pydantic import BaseModel

class HierarchicalClusterRequest(BaseModel):
    model_filename: str
    graph_type: str
    dataset: str