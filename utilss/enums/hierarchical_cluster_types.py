from enum import Enum

class HierarchicalClusterType(Enum):
    SIMILARITY = "similarity"
    DISSIMILARITY = "dissimilarity"
    CONFUSION_MATRIX = "confusion_matrix"
