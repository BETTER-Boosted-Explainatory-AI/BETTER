import tensorflow as tf
import numpy as np
import time
from enums.attack_type import AttackType
from enums.hierarchical_cluster_types import HierarchicalClusterType
from utilss.classes.adversarial_attacks.AdversarialAttack import AdversarialAttack

class DeepFoolAttack(AdversarialAttack):
    """
    DeepFool attack implementation.
    Finds the minimal perturbation needed to change the classification by iteratively
    linearizing the decision boundary.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    
    def __init__(self, model, class_names, Z_full, preprocess_input, 
                 cluster_type=HierarchicalClusterType.SIMILARITY):
        """
        Initialize DeepFool attack.
        
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
        return AttackType.DEEPFOOL
    
    def perturb_images(self, images, num_classes=10, max_iter=20, overshoot=0.02, batch_gradients=True, **kwargs):
        """
        Implements DeepFool attack.
        Works with both CIFAR-100 and ImageNet images.
        
        Args:
            images: Input images (normalized to [0,1])
            num_classes: Number of classes to consider
            max_iter: Maximum iterations
            overshoot: Overshoot parameter
            batch_gradients: Whether to compute gradients in batch (more efficient)
            
        Returns:
            Adversarial images
        """
        # Ensure images are properly normalized
        if np.max(images) > 1.0:
            images = images / 255.0
        
        # Process batch of images
        if len(images.shape) == 4:
            results = []
            for i in range(images.shape[0]):
                adv_image, _, _, _, _ = self.optimized_deepfool(
                    images[i], 
                    num_classes=num_classes, 
                    max_iter=max_iter, 
                    overshoot=overshoot,
                    batch_gradients=batch_gradients
                )
                results.append(adv_image)
            return np.array(results)
        else:
            # Process single image
            adv_image, _, _, _, _ = self.optimized_deepfool(
                images, 
                num_classes=num_classes, 
                max_iter=max_iter, 
                overshoot=overshoot,
                batch_gradients=batch_gradients
            )
            return adv_image
    
    def analyze_attack(self, image_index, original_images, class_names, num_classes=10, max_iter=20, overshoot=0.02, threshold=None):
        """
        Run the DeepFool attack and analyze results with adversarial score.
        
        Parameters:
        - image_index: Index of the image to analyze
        - original_images: Array of original images
        - class_names: List of class names
        - num_classes: Number of classes to consider
        - max_iter: Maximum iterations
        - overshoot: Overshoot parameter
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
        adversarial_image, original_label, adversarial_label, original_probs, adversarial_probs = self.optimized_deepfool(
            original_image,
            num_classes=num_classes,
            max_iter=max_iter,
            overshoot=overshoot
        )
        
        # Calculate adversarial scores
        original_score = self.calculate_adversarial_score(original_probs)
        adversarial_score = self.calculate_adversarial_score(adversarial_probs)
        
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
            'attack_type': 'DEEPFOOL',
            'original_class': original_label,
            'original_class_name': class_names[original_label] if original_label < len(class_names) else 'Unknown',
            'adversarial_class': adversarial_label,
            'adversarial_class_name': class_names[adversarial_label] if adversarial_label < len(class_names) else 'Unknown',
            'attack_success': original_label != adversarial_label,
            'original_score': original_score['score'],
            'original_is_adversarial': original_score['is_adversarial'],
            'adversarial_score': adversarial_score['score'],
            'adversarial_is_adversarial': adversarial_score['is_adversarial'],
            'detection_success': not original_score['is_adversarial'] and adversarial_score['is_adversarial'],
            'l2_norm': float(l2_norm),
            'linf_norm': float(linf_norm),
            'execution_time': execution_time,
            'parameters': {
                'num_classes': num_classes,
                'max_iter': max_iter,
                'overshoot': overshoot,
                'threshold': threshold if threshold is not None else original_score.get('threshold', None)
            }
        }
        
        return results
    
    def optimized_deepfool(self, image, num_classes=10, max_iter=20, overshoot=0.02, batch_gradients=True):
        """
        Optimized memory-efficient implementation of DeepFool.
        Works with both CIFAR-100 and ImageNet images.
        
        Args:
            image: Input image (single image without batch dimension) in [0,1] range
            num_classes: Number of classes to consider for the attack
            max_iter: Maximum number of iterations
            overshoot: Overshoot parameter
            batch_gradients: Whether to compute gradients in batch (faster but more memory)
            
        Returns:
            perturbed_image, original_label, adversarial_label, original_probs, adversarial_probs
        """
        # Make a copy of the image to avoid modifying the original
        x_adv = image.copy()
        
        # Add batch dimension if needed
        if len(x_adv.shape) == 3:
            x_adv_batch = np.expand_dims(x_adv, axis=0)
        else:
            x_adv_batch = x_adv
        
        # Preprocess the image for initial prediction
        x_adv_preprocessed = self.preprocess_input(x_adv_batch * 255.0)
        
        # Get initial prediction
        original_preds = self.model.predict(x_adv_preprocessed, verbose=0)[0]
        original_label = np.argmax(original_preds)
        current_label = original_label
        
        # Convert numpy array to tensor once
        x_tensor = tf.Variable(x_adv_batch, dtype=tf.float32)
        
        # Cache for gradients to avoid recomputation
        gradient_cache = {}
        
        # Loop for max_iter iterations
        for i in range(max_iter):
            # Preprocess the current image
            x_preprocessed = self.preprocess_input(x_tensor.numpy() * 255.0)
            
            # Get current prediction
            current_preds = self.model.predict(x_preprocessed, verbose=0)[0]
            current_label = np.argmax(current_preds)
            
            # Check if we've already succeeded in finding an adversarial example
            if current_label != original_label:
                break
            
            # Initialize variables to track minimum perturbation
            min_dist = float('inf')
            best_pert = None
            target_class = None
            
            # Select top classes by prediction score (excluding current class)
            if num_classes < len(self.class_names):
                top_indices = np.argsort(current_preds)[::-1]
                classes_to_check = [idx for idx in top_indices[:num_classes+1] if idx != current_label][:num_classes]
            else:
                classes_to_check = [c for c in range(len(self.class_names)) if c != current_label]
            
            # Compute gradients in batch if enabled (much faster but uses more memory)
            if batch_gradients:
                grads = self.batch_compute_gradients(x_tensor, current_label, classes_to_check)
            
            # For each target class, find the minimum perturbation
            for k_idx, k in enumerate(classes_to_check):
                # Get gradient either from batch computation or compute individually
                if batch_gradients:
                    grad = grads[k_idx]
                else:
                    # Use cached gradient if available
                    cache_key = f"{current_label}_{k}"
                    if cache_key in gradient_cache:
                        grad = gradient_cache[cache_key]
                    else:
                        grad = self.compute_gradient(x_tensor, current_label, k)
                        gradient_cache[cache_key] = grad
                
                if grad is None:
                    continue
                    
                # Calculate f_k - f_current (loss)
                f_current = current_preds[current_label]
                f_k = current_preds[k]
                loss = f_k - f_current
                
                # Normalize gradient
                grad_norm = np.linalg.norm(grad) + 1e-8
                grad_normalized = grad / grad_norm
                
                # Calculate distance to decision boundary
                dist = abs(loss) / grad_norm
                
                # Check if this is the closest boundary
                if dist < min_dist:
                    min_dist = dist
                    best_pert = grad_normalized
                    target_class = k
            
            # If no valid perturbation found, try fallback strategies
            if best_pert is None:
                # Try harder with FGSM-style update for top classes
                for k in classes_to_check[:5]:  # Try just top 5 classes
                    g = self.fgsm_gradient_optimized(x_tensor, k)
                    if g is not None and np.any(g):
                        g_norm = np.linalg.norm(g) + 1e-8
                        best_pert = g / g_norm
                        target_class = k
                        min_dist = 0.01  # Small fixed step size
                        break
                
                # If still no valid perturbation, use random perturbation as last resort
                if best_pert is None:
                    best_pert = np.random.normal(0, 1, size=x_adv_batch.shape)
                    best_pert = best_pert / (np.linalg.norm(best_pert) + 1e-8)
                    min_dist = 0.01
            
            # Apply perturbation with overshoot
            pert = (1 + overshoot) * min_dist * best_pert
            
            # Update tensor
            x_tensor.assign(tf.clip_by_value(x_tensor + tf.convert_to_tensor(pert, dtype=tf.float32), 0, 1))
            
            # Clear gradient cache at regular intervals to free memory
            if i % 5 == 0:
                gradient_cache.clear()
        
        # Get final prediction
        x_adv = x_tensor.numpy()[0]
        x_adv_preprocessed = self.preprocess_input(np.expand_dims(x_adv * 255.0, axis=0))
        final_preds = self.model.predict(x_adv_preprocessed, verbose=0)[0]
        adversarial_label = np.argmax(final_preds)
        
        return x_adv, original_label, adversarial_label, original_preds, final_preds
    
    def compute_gradient(self, x_tensor, current_class, target_class):
        """
        Compute gradient of (target_class - current_class) with respect to x
        
        Args:
            x_tensor: Input tensor
            current_class: Current predicted class
            target_class: Target class
            
        Returns:
            Gradient
        """
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image
            x_preprocessed = self.preprocess_input(x_tensor * 255.0)
            
            # Get predictions
            preds = self.model(x_preprocessed, training=False)
            
            # Target loss function: maximize target class and minimize current class
            loss = preds[0, target_class] - preds[0, current_class]
        
        # Compute gradient
        try:
            gradient = tape.gradient(loss, x_tensor)
            if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                return None
            return gradient.numpy()
        except Exception as e:
            print(f"Error computing gradient: {e}")
            return None
    
    def batch_compute_gradients(self, x_tensor, current_class, target_classes):
        """
        Compute gradients for multiple target classes in a single batch operation
        
        Args:
            x_tensor: Input tensor
            current_class: Current predicted class
            target_classes: List of target classes
            
        Returns:
            List of gradients for each target class
        """
        gradients = []
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image
            x_preprocessed = self.preprocess_input(x_tensor * 255.0)
            
            # Get predictions
            preds = self.model(x_preprocessed, training=False)
            
            # Store current class prediction
            current_pred = preds[0, current_class]
        
        # Compute gradient for each target class
        for k in target_classes:
            try:
                # Target loss function: maximize target class and minimize current class
                loss = preds[0, k] - current_pred
                
                gradient = tape.gradient(loss, x_tensor)
                if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                    gradients.append(None)
                else:
                    gradients.append(gradient.numpy())
            except Exception as e:
                print(f"Error computing gradient for class {k}: {e}")
                gradients.append(None)
        
        # Free resources
        del tape
                
        return gradients
    
    def fgsm_gradient_optimized(self, x_tensor, target_class):
        """
        Optimized FGSM gradient computation using TensorFlow
        
        Args:
            x_tensor: Input tensor
            target_class: Target class
            
        Returns:
            Gradient
        """
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image
            x_preprocessed = self.preprocess_input(x_tensor * 255.0)
            
            # Get predictions
            preds = self.model(x_preprocessed)
            
            # Target class score
            target_score = preds[0, target_class]
        
        # Compute gradient
        try:
            gradient = tape.gradient(target_score, x_tensor)
            if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                return None
            return gradient.numpy()
        except Exception as e:
            print(f"Error computing FGSM gradient: {e}")
            return None