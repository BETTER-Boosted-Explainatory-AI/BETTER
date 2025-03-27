from enum import Enum, auto

class AttackType(Enum):
    """Enum for supported adversarial attack types"""
    FGSM = auto()
    PGD = auto() 
    DEEPFOOL = auto()