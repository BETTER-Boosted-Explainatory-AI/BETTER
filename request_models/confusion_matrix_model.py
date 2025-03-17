from pydantic import BaseModel
from typing import List

class ConfusionMatrixRequest(BaseModel):
    model_filename: str
    edges_df_filename: str
    dataset: str
