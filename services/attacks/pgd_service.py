from typing import Dict, Any, Optional
import numpy as np
from utilss.enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.enums.attack_type import AttackType
from utilss.classes.adversarial_attacks.PGD import PGDAttack
from utilss.enums.datasets_enum import DatasetsEnum
from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO
import os
import app  # Import app to access global instances

def _get_dataset_config(dataset_str: str) -> Dict[str, Any]:
    """Get dataset configuration based on dataset string."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        return CIFAR100_INFO
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        return IMAGENET_INFO
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def _get_dataset_instance(dataset_str: str):
    """Get the dataset instance using global variables."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        # Use the global CIFAR-100 instance if available
        if app.cifer100_instance is None:
            # If not already loaded, load it
            from utilss.classes.datasets.cifar100 import Cifar100
            app.cifer100_instance = Cifar100()
            app.cifer100_instance.load("cifar100")
        return app.cifer100_instance
        
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        # Use the global ImageNet instance if available
        if app.imagenet_instance is None:
            # If not already loaded, load it
            from utilss.classes.datasets.imagenet import ImageNet
            app.imagenet_instance = ImageNet()
            app.imagenet_instance.load("imagenet")
        return app.imagenet_instance
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")

def _load_model(dataset_str: str):
    """Load the model for the specified dataset."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        # Load CIFAR-100 model
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(weights=None, classes=100)
        # Load model weights - adjust the path as needed
        model_path = os.path.join(app.FOLDER_PATH, "models", "cifar100_model.h5")
        model.load_weights(model_path)
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        # Load ImageNet model
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet')
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")
        
    return model, preprocess_input

def _get_hierarchical_clustering_data(dataset_str: str):
    """Get hierarchical clustering data for the specified dataset."""
    if dataset_str == DatasetsEnum.CIFAR100.value:
        # Load CIFAR-100 hierarchical clustering data
        file_path = os.path.join(app.FOLDER_PATH, "data", "cifar100_hierarchical_clustering.npy")
        Z_full = np.load(file_path)
    elif dataset_str == DatasetsEnum.IMAGENET.value:
        # Load ImageNet hierarchical clustering data
        file_path = os.path.join(app.FOLDER_PATH, "data", "imagenet_hierarchical_clustering.npy")
        Z_full = np.load(file_path)
    else:
        raise ValueError(f"Invalid dataset: {dataset_str}")
        
    return Z_full

def perform_pgd_attack(
    dataset_name: str,
    image_index: int,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_steps: int = 40,
    targeted: bool = False,
    target_class: Optional[int] = None,
    threshold: Optional[float] = None,
    cluster_type: HierarchicalClusterType = HierarchicalClusterType.SIMILARITY
):
    """
    Perform PGD attack on the specified image.
    
    Args:
        dataset_name: Name of the dataset
        image_index: Index of the image to attack
        epsilon: Maximum perturbation size
        alpha: Step size for each iteration
        num_steps: Number of attack iterations
        targeted: Whether to perform a targeted attack
        target_class: Target class for targeted attacks
        threshold: Custom threshold for adversarial detection
        cluster_type: Type of hierarchical clustering to use
        
    Returns:
        Dictionary with attack results
    """
    # Get dataset configuration
    dataset_config = _get_dataset_config(dataset_name)
    
    # Get dataset instance
    dataset = _get_dataset_instance(dataset_name)
    
    # Load model and preprocessing function
    model, preprocess_input = _load_model(dataset_name)
    
    # Get class names
    class_names = dataset_config["labels"]
    
    # Get hierarchical clustering data
    Z_full = _get_hierarchical_clustering_data(dataset_name)
    
    # Create PGD attack instance
    pgd_attack = PGDAttack(
        model=model,
        class_names=class_names,
        Z_full=Z_full,
        preprocess_input=preprocess_input,
        cluster_type=cluster_type
    )
    
    # Get original images based on dataset type
    if dataset_name == DatasetsEnum.CIFAR100.value:
        # For CIFAR-100, use the test set
        original_images = dataset.x_test
    elif dataset_name == DatasetsEnum.IMAGENET.value:
        # For ImageNet, extract images from test_data
        # Adapt this to match your ImageNet class implementation
        original_images = np.array([img for img, _ in dataset.test_data])
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    
    # Perform the attack analysis
    if targeted and target_class is not None:
        # If targeted attack is requested
        # First run normal analysis
        results = pgd_attack.analyze_attack(
            image_index=image_index,
            original_images=original_images,
            class_names=class_names,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            threshold=threshold
        )
        
        # Then perform targeted attack
        # Get the original image
        original_image = original_images[image_index].astype(np.float32)
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0
            
        # Generate targeted adversarial example
        targeted_adversarial_image = pgd_attack.pgd_attack_targeted(
            original_image,
            target_class=target_class,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps
        )
        
        # Add targeted attack results to the output
        results["targeted_attack"] = {
            "target_class": target_class,
            "target_class_name": class_names[target_class] if target_class < len(class_names) else "Unknown"
        }
    else:
        # Perform normal (untargeted) attack analysis
        results = pgd_attack.analyze_attack(
            image_index=image_index,
            original_images=original_images,
            class_names=class_names,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            threshold=threshold
        )
    
    return results