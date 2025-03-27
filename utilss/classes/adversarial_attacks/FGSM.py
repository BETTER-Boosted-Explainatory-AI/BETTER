import tensorflow as tf
import numpy as np
import time
from enums.attack_type import AttackType
from enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.classes.adversarial_attacks.AdversarialAttack import AdversarialAttack
from scoring import Scoring

class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method attack implementation.
    FGSM is a one-step attack that perturbs an image in the direction of the gradient
    of the loss with respect to the input.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    
    def __init__(self, model, class_names, Z_full, preprocess_input, 
                 cluster_type=HierarchicalClusterType.SIMILARITY):
        """
        Initialize FGSM attack.
        
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
    
    def get_attack_type(self):
        """
        Returns the attack type enum value.
        """
        return AttackType.FGSM
    
    def get_default_threshold(self):
        """
        Get the default threshold for this attack and cluster type.
        
        Returns:
            Default threshold value
        """
        # Default thresholds based on empirical evaluation
        if self.cluster_type == HierarchicalClusterType.SIMILARITY:
            return 120  # Default for FGSM similarity graphs
        else:
            return 230  # Default for FGSM distance-based graphs
    
    def perturb_images(self, images, epsilon=0.1, targeted=False, target_class=None, **kwargs):
        """
        Implements Fast Gradient Sign Method (FGSM) attack.
        Works with both CIFAR-100 and ImageNet images.
        
        Args:
            images: Input images (normalized to [0,1])
            epsilon: Attack strength parameter
            targeted: Whether to perform a targeted attack
            target_class: Target class for targeted attacks (int or one-hot encoded)
            
        Returns:
            Adversarial images
        """
        # Ensure images are properly normalized
        if np.max(images) > 1.0:
            images = images / 255.0
        
        # Ensure images have batch dimension
        if len(images.shape) == 3:
            images_batch = np.expand_dims(images, axis=0)
        else:
            images_batch = images
        
        # Convert to tensor
        images_tensor = tf.convert_to_tensor(images_batch, dtype=tf.float32)
        
        # For untargeted attacks, target is the original class
        if not targeted or target_class is None:
            # Get original predictions
            images_preprocessed = self.preprocess_input(images_tensor * 255.0)
            original_preds = self.model(images_preprocessed, training=False)
            original_classes = tf.argmax(original_preds, axis=1)
            
            # Create targets (one-hot encoding of original predictions)
            targets = tf.one_hot(original_classes, depth=len(self.class_names))
        else:
            # For targeted attacks, use the specified target class
            if isinstance(target_class, (int, np.integer)):
                # If target_class is an integer, convert to one-hot
                targets = tf.one_hot([target_class] * images_batch.shape[0], depth=len(self.class_names))
            else:
                # Assume target_class is already one-hot encoded
                targets = target_class
        
        # Compute gradient and create perturbation
        with tf.GradientTape() as tape:
            tape.watch(images_tensor)
            # Preprocess images for the model
            images_preprocessed = self.preprocess_input(images_tensor * 255.0)
            predictions = self.model(images_preprocessed, training=False)
            
            # For targeted attacks, minimize loss between predictions and target
            # For untargeted attacks, maximize loss
            loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
            if targeted:
                loss = -loss
        
        # Get the gradients
        gradients = tape.gradient(loss, images_tensor)
        
        # Create the perturbation using the sign of gradients
        perturbation = epsilon * tf.sign(gradients)
        
        # Add perturbation to images
        adversarial_images = images_tensor + perturbation
        
        # Clip to ensure valid image range [0, 1]
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
        
        # If input was a single image, return a single image
        if len(images.shape) == 3:
            return adversarial_images.numpy()[0]
        return adversarial_images.numpy()
    
    def analyze_attack(self, image_index, original_images, class_names, epsilon=0.1, targeted=False, target_class=None, threshold=None):
        """
        Run the FGSM attack and analyze results with adversarial score.
        
        Parameters:
        - image_index: Index of the image to analyze
        - original_images: Array of original images
        - class_names: List of class names
        - epsilon: Attack strength parameter
        - targeted: Whether to perform a targeted attack
        - target_class: Target class for targeted attacks
        - threshold: Custom threshold for adversarial detection (uses default if None)
        
        Returns:
        - Dictionary with analysis results
        """
        start_time = time.time()
        
        # Use default threshold if none provided
        if threshold is None:
            threshold = self.get_default_threshold()
        
        # Get the original image
        original_image = original_images[image_index].astype(np.float32)
        
        # Normalize to [0, 1] range for the attack
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0
        
        # Generate adversarial example
        adversarial_image = self.perturb_images(
            original_image,
            epsilon=epsilon,
            targeted=targeted,
            target_class=target_class
        )
        
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
        
        # Calculate adversarial scores using the Scoring class
        original_score = self.scorer.calculate_adversarial_score(original_preds[0])
        adversarial_score = self.scorer.calculate_adversarial_score(adversarial_preds[0])
        
        # Determine if adversarial using the custom threshold
        original_is_adversarial = self.scorer.is_adversarial(original_preds[0], threshold)
        adversarial_is_adversarial = self.scorer.is_adversarial(adversarial_preds[0], threshold)
        
        # Calculate L2 and L-infinity norms of the perturbation
        l2_norm = np.linalg.norm((adversarial_image - original_image).flatten())
        linf_norm = np.max(np.abs(adversarial_image - original_image))
        
        # Execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            'attack_type': 'FGSM',
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
            'parameters': {
                'epsilon': epsilon,
                'targeted': targeted,
                'target_class': target_class if targeted else None,
                'threshold': threshold
            }
        }
        
        return results
    
    def run_evaluation(self, original_images, true_labels, class_names, num_samples=100, epsilon=0.1, threshold=None):
        """
        Run FGSM attack evaluation on multiple samples.
        
        Parameters:
        - original_images: Array of original images
        - true_labels: Array of true labels (can be None if not needed)
        - class_names: List of class names
        - num_samples: Number of samples to evaluate
        - epsilon: Attack strength parameter
        - threshold: Custom threshold for adversarial detection (uses default if None)
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Use default threshold if none provided
        if threshold is None:
            threshold = self.get_default_threshold()
            
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
            adversarial_image = self.perturb_images(original_image, epsilon=epsilon)
            
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
            original_score = self.scorer.calculate_adversarial_score(original_preds[0])
            adversarial_score = self.scorer.calculate_adversarial_score(adversarial_preds[0])
            
            # Determine if adversarial using the custom threshold
            original_is_adversarial = self.scorer.is_adversarial(original_preds[0], threshold)
            adversarial_is_adversarial = self.scorer.is_adversarial(adversarial_preds[0], threshold)
            
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
            'epsilon': epsilon,
            'num_samples': num_samples
        }
        
        return results
    
    def fgsm_attack_for_graph(self, images, targets, epsilon=0.01, clip_min=-123, clip_max=151):
        """
        Special version of FGSM optimized for graph analysis with specific clipping values.
        Supports both CIFAR-100 and ImageNet preprocessing.
        
        Args:
            images: Input images (already preprocessed)
            targets: Target labels (one-hot encoded)
            epsilon: Perturbation magnitude
            clip_min: Minimum clipping value for preprocessed images
            clip_max: Maximum clipping value for preprocessed images
            
        Returns:
            Adversarial examples (preprocessed)
        """
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        else:
            images_tensor = images
            
        # Create a variable to track gradients
        images_var = tf.Variable(images_tensor)
        
        with tf.GradientTape() as tape:
            # The model expects preprocessed input
            predictions = self.model(images_var)
            # For untargeted attack: maximize loss for true class
            loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
        
        # Get the gradients
        gradients = tape.gradient(loss, images_var)
        
        # Create perturbation (sign of gradient)
        perturbation = epsilon * tf.sign(gradients)
        
        # Add perturbation to create adversarial example
        adversarial_images = images_tensor + perturbation
        
        # Clip to ensure valid range with specific bounds
        # This is for preprocessed images with specific range
        adversarial_images = tf.clip_by_value(
            adversarial_images, 
            images_tensor - epsilon,  # Lower bound: original - epsilon
            images_tensor + epsilon   # Upper bound: original + epsilon
        )
        
        # Additional global clipping if needed
        if clip_min is not None and clip_max is not None:
            adversarial_images = tf.clip_by_value(adversarial_images, clip_min, clip_max)
        
        return adversarial_images.numpy()