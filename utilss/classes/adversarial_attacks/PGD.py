import tensorflow as tf
import numpy as np
import time
from enums.attack_type import AttackType
from enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.classes.adversarial_attacks.AdversarialAttack import AdversarialAttack

class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent attack implementation.
    PGD is one of the strongest first-order attacks, iteratively taking gradient
    steps and projecting back onto an ε-ball around the original image.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    
    def __init__(self, model, class_names, Z_full, preprocess_input, 
                 cluster_type=HierarchicalClusterType.SIMILARITY):
        """
        Initialize PGD attack.
        
        Args:
            model: The model to attack
            class_names: List of class names
            Z_full: The hierarchical clustering data
            preprocess_input: Function to preprocess inputs for the model
            cluster_type: Type of clustering (SIMILARITY, DISSIMILARITY, CONFUSION_MATRIX)
        """
        super().__init__(model, class_names, Z_full, preprocess_input, cluster_type)
    
    def get_attack_type(self):
        """
        Returns the attack type enum value.
        """
        return AttackType.PGD
    
    def perturb_images(self, images, epsilon=0.1, alpha=0.01, num_steps=40, targeted=False, target_class=None, **kwargs):
        """
        Implements Projected Gradient Descent (PGD) attack.
        Works with both CIFAR-100 and ImageNet images.
        
        Args:
            images: Input images (normalized to [0,1])
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
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
            
        # Convert to tensor and make it a variable
        images_tensor = tf.Variable(images_batch, dtype=tf.float32)
        original_images = tf.identity(images_tensor)
        
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
        
        # PGD attack loop
        for step in range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(images_tensor)
                # Preprocess images for the model
                images_preprocessed = self.preprocess_input(images_tensor * 255.0)
                predictions = self.model(images_preprocessed, training=False)
                
                # Use categorical crossentropy loss
                loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
                # For targeted attacks, minimize loss instead of maximizing
                if targeted:
                    loss = -loss
            
            # Get the gradients
            gradients = tape.gradient(loss, images_tensor)
            
            # Create the perturbation using the sign of gradients
            perturbation = alpha * tf.sign(gradients)
            
            # Add perturbation to images
            images_tensor.assign_add(perturbation)
            
            # Project back to epsilon ball around original image
            delta = tf.clip_by_value(images_tensor - original_images, -epsilon, epsilon)
            images_tensor.assign(original_images + delta)
            
            # Clip to ensure valid image range [0, 1]
            images_tensor.assign(tf.clip_by_value(images_tensor, 0.0, 1.0))
        
        # If input was a single image, return a single image
        if len(images.shape) == 3:
            return images_tensor.numpy()[0]
        return images_tensor.numpy()
    
    def analyze_attack(self, image_index, original_images, class_names, epsilon=0.1, alpha=0.01, num_steps=40, threshold=None):
        """
        Run the PGD attack and analyze results with adversarial score.
        
        Parameters:
        - image_index: Index of the image to analyze
        - original_images: Array of original images
        - class_names: List of class names
        - epsilon: Maximum perturbation size
        - alpha: Step size
        - num_steps: Number of attack iterations
        - threshold: Optional custom threshold for adversarial detection
        
        Returns:
        - Dictionary with analysis results
        """
        start_time = time.time()
        
        # Get the original image
        original_image = original_images[image_index].astype(np.float32)
        
        # Normalize to [0, 1] range for the attack
        if np.max(original_image) > 1.0:
            original_image = original_image / 255.0
        
        # Generate adversarial example
        adversarial_image = self.perturb_images(
            original_image,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps
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
        
        # Calculate adversarial scores
        original_score = self.calculate_adversarial_score(original_preds[0])
        adversarial_score = self.calculate_adversarial_score(adversarial_preds[0])
        
        # If custom threshold provided, override the is_adversarial determination
        if threshold is not None:
            original_score['is_adversarial'] = original_score['score'] > threshold
            adversarial_score['is_adversarial'] = adversarial_score['score'] > threshold
        
        # Calculate L2 and L-infinity norms of the perturbation
        l2_norm = np.linalg.norm((adversarial_image - original_image).flatten())
        linf_norm = np.max(np.abs(adversarial_image - original_image))
        
        # Execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            'attack_type': 'PGD',
            'original_class': original_class,
            'original_class_name': class_names[original_class] if original_class < len(class_names) else 'Unknown',
            'adversarial_class': adversarial_class,
            'adversarial_class_name': class_names[adversarial_class] if adversarial_class < len(class_names) else 'Unknown',
            'attack_success': original_class != adversarial_class,
            'original_score': original_score['score'],
            'original_is_adversarial': original_score['is_adversarial'],
            'adversarial_score': adversarial_score['score'],
            'adversarial_is_adversarial': adversarial_score['is_adversarial'],
            'detection_success': not original_score['is_adversarial'] and adversarial_score['is_adversarial'],
            'l2_norm': float(l2_norm),
            'linf_norm': float(linf_norm),
            'execution_time': execution_time,
            'parameters': {
                'epsilon': epsilon,
                'alpha': alpha,
                'num_steps': num_steps,
                'threshold': threshold if threshold is not None else original_score.get('threshold', None)
            }
        }
        
        return results
    
    def pgd_attack_targeted(self, images, target_class, epsilon=0.1, alpha=0.01, num_steps=40):
        """
        Targeted PGD attack that attempts to force classification as a specific target class.
        Works with both CIFAR-100 and ImageNet images.
        
        Args:
            images: Input images (normalized to [0,1])
            target_class: Target class index or one-hot encoded target
            epsilon: Maximum perturbation size
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            
        Returns:
            Adversarial images
        """
        return self.perturb_images(
            images, 
            epsilon=epsilon, 
            alpha=alpha, 
            num_steps=num_steps,
            targeted=True,
            target_class=target_class
        )