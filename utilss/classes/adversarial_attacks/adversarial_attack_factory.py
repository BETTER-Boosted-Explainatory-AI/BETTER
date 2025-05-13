from utilss.classes.adversarial_attacks.PGD import PGDAttack
from utilss.classes.adversarial_attacks.FGSM import FGSMAttack
from utilss.classes.adversarial_attacks.DeepFool import DeepFoolAttack
from utilss.enums.attack_type import AttackType

def get_attack(attack_name: str, **kwargs):
    """
    Factory function to get the appropriate attack class based on the attack name.

    Args:
        attack_name (AttackType): Enum value representing the attack type.

    Returns:
        AdversarialAttack: An instance of the corresponding attack class.
    """
    attack_name = attack_name.lower()
    
    attack_classes = {
        AttackType.PGD.value: PGDAttack,
        AttackType.FGSM.value: FGSMAttack,
        AttackType.DEEPFOOL.value: DeepFoolAttack
    }

    if attack_name not in attack_classes:
        raise ValueError(f"Unknown attack name: {attack_name}")
    
    # Handle special cases where additional arguments are required
    if attack_name == AttackType.DEEPFOOL.value:
        if "class_names" not in kwargs:
            raise ValueError("DeepFoolAttack requires 'class_names' as an argument.")
        return attack_classes[attack_name](kwargs["class_names"])

    return attack_classes[attack_name]()