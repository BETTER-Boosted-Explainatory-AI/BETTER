from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time

from utilss.enums.attack_type import AttackType
from utilss.enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.classes.scoring import Scoring

class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks.
    Defines the interface that all specific attack implementations must follow.
    Focuses on core attack functionality and attack-specific analysis.
    """
    
    def __init__(self, model, class_names, Z_full, preprocess_input, 
                 cluster_type=HierarchicalClusterType.SIMILARITY):
        """
        Initialize the attack.
        
        Args:
            model: The model to attack
            class_names: List of class names
            Z_full: The hierarchical clustering data
            preprocess_input: Function to preprocess inputs for the model
            cluster_type: Type of clustering (SIMILARITY, DISSIMILARITY, CONFUSION_MATRIX)
        """
        self.model = model
        self.class_names = class_names
        self.Z_full = Z_full
        self.preprocess_input = preprocess_input
        self.cluster_type = cluster_type
        
        # Create scorer instance
        self.scorer = Scoring(Z_full, class_names)
    
    @abstractmethod
    def perturb_images(self, images, **kwargs):
        """
        Generate adversarial examples from input images.
        
        Args:
            images: Input images to perturb
            **kwargs: Attack-specific parameters
            
        Returns:
            Adversarial examples
        """
        pass
    
    @abstractmethod
    def get_attack_type(self):
        """
        Returns the attack type enum value.
        """
        pass
    
    
    def calculate_adversarial_score(self, predictions, top_k=5):
        """
        Calculate adversarial score based on hierarchical clustering data.
        
        Args:
            predictions: Model predictions
            top_k: Number of top predictions to consider
            
        Returns:
            Dictionary with adversarial score and detection
        """
        return self.scorer.calculate_adversarial_score(predictions, top_k)
    
    def is_adversarial(self, predictions, threshold=None, top_k=5):
        """
        Quick check if predictions are adversarial based on score.
        
        Parameters:
        - predictions: Model output predictions
        - threshold: Score threshold for adversarial detection (uses default if None)
        - top_k: Number of top predictions to consider
        
        Returns:
        - Boolean indicating if the prediction is adversarial
        """
        if threshold is None:
            threshold = self.get_default_threshold()
            
        return self.scorer.is_adversarial(predictions, threshold, top_k)
    
    def vulnerability_check(self, images, **kwargs):
        """
        Check model vulnerability to this attack.
        
        Args:
            images: Input images to test
            **kwargs: Attack-specific parameters
            
        Returns:
            Dictionary with vulnerability metrics
        """
        start_time = time.time()
        
        # Use default threshold if none provided
        threshold = kwargs.pop('threshold', self.get_default_threshold())
        
        # Normalize images to [0,1] if needed
        if np.max(images) > 1.0:
            images = images / 255.0
        
        # Get original predictions
        if len(images.shape) == 3:
            images_batch = np.expand_dims(images, axis=0)
        else:
            images_batch = images
            
        orig_preprocessed = self.preprocess_input(images_batch * 255.0)
        orig_preds = self.model.predict(orig_preprocessed, verbose=0)
        orig_classes = np.argmax(orig_preds, axis=1)
        
        # Generate adversarial examples
        adv_images = self.perturb_images(images, **kwargs)
        
        # Get adversarial predictions
        adv_preprocessed = self.preprocess_input(adv_images * 255.0 if len(adv_images.shape) > 3 else np.expand_dims(adv_images, axis=0) * 255.0)
        adv_preds = self.model.predict(adv_preprocessed, verbose=0)
        adv_classes = np.argmax(adv_preds, axis=1)
        
        # Calculate success rate
        success_rate = np.mean(orig_classes != adv_classes)
        
        # Calculate adversarial scores
        orig_scores = [self.calculate_adversarial_score(pred)['score'] for pred in orig_preds]
        adv_scores = [self.calculate_adversarial_score(pred)['score'] for pred in adv_preds]
        
        # Calculate average scores and detection rate
        avg_orig_score = np.mean(orig_scores)
        avg_adv_score = np.mean(adv_scores)
        
        # Use the specified threshold for detection
        orig_detected = [score > threshold for score in orig_scores]
        adv_detected = [score > threshold for score in adv_scores]
        
        detection_rate = np.mean(np.array(adv_detected) & ~np.array(orig_detected))
        false_positive_rate = np.mean(orig_detected)
        
        elapsed_time = time.time() - start_time
        
        return {
            'attack_type': self.get_attack_type().name,
            'success_rate': float(success_rate),
            'detection_rate': float(detection_rate),
            'false_positive_rate': float(false_positive_rate),
            'avg_original_score': float(avg_orig_score),
            'avg_adversarial_score': float(avg_adv_score),
            'elapsed_time': elapsed_time,
            'adversarial_images': adv_images,
            'original_classes': orig_classes,
            'adversarial_classes': adv_classes,
            'threshold': threshold
        }
    
    def consistency_check(self, model, images, **kwargs):
        """
        Check consistency of model predictions under different attack parameters.
        
        Args:
            model: Model to evaluate
            images: Input images to test
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with consistency metrics
        """
        # Get original predictions
        if len(images.shape) == 3:
            images_batch = np.expand_dims(images, axis=0)
        else:
            images_batch = images
            
        orig_preprocessed = self.preprocess_input(images_batch * 255.0)
        orig_preds = model.predict(orig_preprocessed, verbose=0)
        orig_classes = np.argmax(orig_preds, axis=1)
        
        attack_type = self.get_attack_type()
        results = {}
        
        # Test different attack parameters based on attack type
        if attack_type == AttackType.FGSM:
            epsilons = kwargs.get('epsilons', [0.01, 0.05, 0.1, 0.2])
            for epsilon in epsilons:
                # Generate adversarial examples with this epsilon
                adv_images = self.perturb_images(images, epsilon=epsilon)
                
                # Get predictions
                adv_preprocessed = self.preprocess_input(adv_images * 255.0 if len(adv_images.shape) > 3 else np.expand_dims(adv_images, axis=0) * 255.0)
                adv_preds = model.predict(adv_preprocessed, verbose=0)
                adv_classes = np.argmax(adv_preds, axis=1)
                
                # Calculate success rate
                success_rate = np.mean(orig_classes != adv_classes)
                
                results[f'epsilon_{epsilon}'] = {
                    'success_rate': float(success_rate),
                    'num_changed': int(np.sum(orig_classes != adv_classes))
                }
                
        elif attack_type == AttackType.PGD:
            # Test different iterations and step sizes
            num_steps_list = kwargs.get('num_steps_list', [10, 20, 40])
            alpha_list = kwargs.get('alpha_list', [0.005, 0.01, 0.02])
            epsilon = kwargs.get('epsilon', 0.1)
            
            for num_steps in num_steps_list:
                for alpha in alpha_list:
                    # Generate adversarial examples
                    adv_images = self.perturb_images(images, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
                    
                    # Get predictions
                    adv_preprocessed = self.preprocess_input(adv_images * 255.0 if len(adv_images.shape) > 3 else np.expand_dims(adv_images, axis=0) * 255.0)
                    adv_preds = model.predict(adv_preprocessed, verbose=0)
                    adv_classes = np.argmax(adv_preds, axis=1)
                    
                    # Calculate success rate
                    success_rate = np.mean(orig_classes != adv_classes)
                    
                    results[f'steps_{num_steps}_alpha_{alpha}'] = {
                        'success_rate': float(success_rate),
                        'num_changed': int(np.sum(orig_classes != adv_classes))
                    }
                    
        elif attack_type == AttackType.DEEPFOOL:
            # Test different numbers of iterations and classes
            max_iter_list = kwargs.get('max_iter_list', [5, 10, 20])
            num_classes_list = kwargs.get('num_classes_list', [5, 10, 20])
            
            for max_iter in max_iter_list:
                for num_classes in num_classes_list:
                    # Generate adversarial examples
                    adv_images = self.perturb_images(images, num_classes=num_classes, max_iter=max_iter)
                    
                    # Get predictions
                    adv_preprocessed = self.preprocess_input(adv_images * 255.0 if len(adv_images.shape) > 3 else np.expand_dims(adv_images, axis=0) * 255.0)
                    adv_preds = model.predict(adv_preprocessed, verbose=0)
                    adv_classes = np.argmax(adv_preds, axis=1)
                    
                    # Calculate success rate
                    success_rate = np.mean(orig_classes != adv_classes)
                    
                    results[f'iter_{max_iter}_classes_{num_classes}'] = {
                        'success_rate': float(success_rate),
                        'num_changed': int(np.sum(orig_classes != adv_classes))
                    }
        
        return results
    
    def analyze_attack(self, image_index, original_images, class_names, **kwargs):
        """
        Run the attack and analyze results with adversarial score.
        This is a base method that should be overridden by specific attack implementations.
        
        Parameters:
        - image_index: Index of the image to analyze
        - original_images: Array of original images
        - class_names: List of class names
        - **kwargs: Attack-specific parameters
        
        Returns:
        - Dictionary with analysis results
        """
        threshold = kwargs.pop('threshold', self.get_default_threshold())
        
        start_time = time.time()
        
        # Get the original image
        original_image = original_images[image_index].astype(np.float32)
        
        # Normalize to [0, 1] range for the attack
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0
        
        # Generate adversarial example
        adversarial_image = self.perturb_images(original_image, **kwargs)
        
        # Convert to batch format for prediction
        original_image_batch = np.expand_dims(original_image, axis=0)
        adversarial_image_batch = np.expand_dims(adversarial_image, axis=0)
        
        # Preprocess for model prediction
        original_preprocessed = self.preprocess_input(original_image_batch * 255.0)
        adversarial_preprocessed = self.preprocess_input(adversarial_image_batch * 255.0)
        
        # Get predictions
        original_preds = self.model.predict(original_preprocessed, verbose=0)
        adversarial_preds = self.model.predict(adversarial_preprocessed, verbose=0)
        
        # Get predicted classes
        original_class = np.argmax(original_preds[0])
        adversarial_class = np.argmax(adversarial_preds[0])
        
        # Calculate adversarial scores
        original_score = self.calculate_adversarial_score(original_preds[0])
        adversarial_score = self.calculate_adversarial_score(adversarial_preds[0])
        
        # Determine if adversarial using the specified threshold
        original_is_adversarial = self.is_adversarial(original_preds[0], threshold)
        adversarial_is_adversarial = self.is_adversarial(adversarial_preds[0], threshold)
        
        # Calculate L2 and L-infinity norms of the perturbation
        l2_norm = np.linalg.norm((adversarial_image - original_image).flatten())
        linf_norm = np.max(np.abs(adversarial_image - original_image))
        
        # Execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            'attack_type': self.get_attack_type().name,
            'original_class': original_class,
            'original_class_name': class_names[original_class] if original_class < len(class_names) else 'Unknown',
            'adversarial_class': adversarial_class,
            'adversarial_class_name': class_names[adversarial_class] if adversarial_class < len(class_names) else 'Unknown',
            'attack_success': original_class != adversarial_class,
            'original_score': original_score['score'],
            'original_is_adversarial': original_is_adversarial,
            'adversarial_score': adversarial_score['score'],
            'adversarial_is_adversarial': adversarial_is_adversarial,
            'detection_success': not original_is_adversarial and adversarial_is_adversarial,
            'l2_norm': float(l2_norm),
            'linf_norm': float(linf_norm),
            'execution_time': execution_time,
            'parameters': kwargs,
            'threshold': threshold
        }
        
        return results
    
    def run_evaluation(self, original_images, true_labels, class_names, num_samples=100, **kwargs):
        """
        Run attack evaluation on multiple samples.
        This is a base method that should be overridden by specific attack implementations.
        
        Parameters:
        - original_images: Array of original images
        - true_labels: Array of true labels (can be None if not used)
        - class_names: List of class names
        - num_samples: Number of samples to evaluate
        - **kwargs: Attack-specific parameters
        
        Returns:
        - Dictionary with evaluation metrics
        """
        threshold = kwargs.pop('threshold', self.get_default_threshold())
        
        detected = 0
        successful_attacks = 0
        false_positives = 0
        scores_original = []
        scores_adversarial = []
        
        # Limit number of samples to evaluate
        num_samples = min(num_samples, len(original_images))
        
        for i in range(num_samples):
            # Get original image
            original_image = original_images[i].astype(np.float32)
            if np.max(original_image) > 1.0:
                original_image = original_image / 255.0
            
            # Generate adversarial example
            adversarial_image = self.perturb_images(original_image, **kwargs)
            
            # Convert to batch format for prediction
            original_image_batch = np.expand_dims(original_image, axis=0)
            adversarial_image_batch = np.expand_dims(adversarial_image, axis=0)
            
            # Preprocess for model prediction
            original_preprocessed = self.preprocess_input(original_image_batch * 255.0)
            adversarial_preprocessed = self.preprocess_input(adversarial_image_batch * 255.0)
            
            # Get predictions
            original_preds = self.model.predict(original_preprocessed, verbose=0)
            adversarial_preds = self.model.predict(adversarial_preprocessed, verbose=0)
            
            # Get predicted classes
            original_class = np.argmax(original_preds[0])
            adversarial_class = np.argmax(adversarial_preds[0])
            
            # Calculate adversarial scores
            original_score = self.calculate_adversarial_score(original_preds[0])
            adversarial_score = self.calculate_adversarial_score(adversarial_preds[0])
            
            # Determine if adversarial using the specified threshold
            original_is_adversarial = self.is_adversarial(original_preds[0], threshold)
            adversarial_is_adversarial = self.is_adversarial(adversarial_preds[0], threshold)
            
            # Store scores
            scores_original.append(original_score['score'])
            scores_adversarial.append(adversarial_score['score'])
            
            # Check if attack was successful
            if adversarial_class != original_class:
                successful_attacks += 1
            
            # Count detections
            if not original_is_adversarial and adversarial_is_adversarial:
                detected += 1
            elif original_is_adversarial and not (adversarial_class != original_class):
                false_positives += 1
        
        # Calculate metrics
        detection_rate = detected / num_samples if num_samples > 0 else 0
        success_rate = successful_attacks / num_samples if num_samples > 0 else 0
        false_positive_rate = false_positives / num_samples if num_samples > 0 else 0
        avg_original_score = sum(scores_original) / len(scores_original) if scores_original else 0
        avg_adversarial_score = sum(scores_adversarial) / len(scores_adversarial) if scores_adversarial else 0
        
        results = {
            'detection_rate': float(detection_rate),
            'attack_success_rate': float(success_rate),
            'false_positive_rate': float(false_positive_rate),
            'avg_original_score': float(avg_original_score),
            'avg_adversarial_score': float(avg_adversarial_score),
            'threshold': threshold,
            'num_samples': num_samples,
            'parameters': kwargs
        }
        
        return results