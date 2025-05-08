from enums.attack_type import AttackType
from enums.datasets_enum import DatasetsEnum
from enums.hierarchical_cluster_types import HierarchicalClusterType

from FGSM import FGSMAttack
from PGD import PGDAttack
from DeepFool import DeepFoolAttack
from utilss.classes.score_calculator import Scoring

num_samples = 50

class AttacksManager:
    """
    Main class to manage adversarial attacks for multiple datasets and graph types.
    Orchestrates high-level evaluation, analysis, and visualization of attacks.
    """
    
    def __init__(self, model_id, preprocess_input, class_names, Z_full, dataset_type=DatasetsEnum.CIFAR100, 
                 cluster_type=HierarchicalClusterType.SIMILARITY, og_images=None, perturbed_images=None,
                 threshold_map=None):
        """
        Initialize AttacksManager.
        
        Args:
            model_id: Identifier for the model
            preprocess_input: Function to preprocess inputs for the model
            class_names: List of class names
            Z_full: The hierarchical clustering data
            dataset_type: Type of dataset (CIFAR100 or IMAGENET)
            cluster_type: Type of clustering (SIMILARITY, DISSIMILARITY, CONFUSION_MATRIX)
            og_images: Original images (optional)
            perturbed_images: Already perturbed images (optional)
            threshold_map: Dictionary mapping (dataset_type, cluster_type, attack_type) to thresholds
        """
        self.model_id = model_id
        self.preprocess_input = preprocess_input
        self.class_names = class_names
        self.Z_full = Z_full
        self.dataset_type = dataset_type
        self.cluster_type = cluster_type
        self.og_images = og_images
        self.perturbed_images = perturbed_images
        self.threshold_map = threshold_map or self._get_default_thresholds()
        
        # Create scoring instance
        self.scorer = Scoring(Z_full, class_names)
        
        # Model will be loaded when needed
        self.model = None
        
        # Initialize attack instances dictionary (will be populated when model is loaded)
        self.attacks = {}

    
    def get_threshold(self, attack_type):
        """
        Get the appropriate threshold for the current dataset, clustering type, and attack.
        
        Args:
            attack_type: The attack type to get threshold for
            
        Returns:
            Threshold value
        """
        key = (self.dataset_type, self.cluster_type, attack_type)
        return self.threshold_map.get(key, 120)  # Default fallback threshold
    
    def load_model(self):
        """
        Load the model based on model_id and dataset_type
        """
        # Only load if not already loaded
        if self.model is None:
            # Create a Model instance using your existing Model class
            from utilss.classes.model import Model
            
            # Determine appropriate parameters based on dataset
            if self.dataset_type == DatasetsEnum.CIFAR100:
                dataset_name = "cifar100"
                top_k = 5
            else:  # IMAGENET
                dataset_name = "imagenet"
                top_k = 5
                
            model_instance = Model(
                model=None,  # Will be loaded from file
                top_k=top_k,
                min_confidence=0.1,  # Default value
                model_filename=self.model_id,
                dataset=dataset_name
            )
            
            # Load the model using your existing method
            model_instance.load_model()
            self.model = model_instance.model
            
            # Initialize attack instances after model is loaded
            self._initialize_attacks()
    
    def _initialize_attacks(self):
        """
        Initialize attack instances with appropriate settings
        """
        if self.model is None:
            raise ValueError("Model must be loaded before initializing attacks")
            
        self.attacks = {
            AttackType.FGSM: FGSMAttack(
                model=self.model, 
                class_names=self.class_names, 
                Z_full=self.Z_full, 
                preprocess_input=self.preprocess_input,
                cluster_type=self.cluster_type
            ),
            AttackType.PGD: PGDAttack(
                model=self.model, 
                class_names=self.class_names, 
                Z_full=self.Z_full, 
                preprocess_input=self.preprocess_input,
                cluster_type=self.cluster_type
            ),
            AttackType.DEEPFOOL: DeepFoolAttack(
                model=self.model, 
                class_names=self.class_names, 
                Z_full=self.Z_full, 
                preprocess_input=self.preprocess_input,
                cluster_type=self.cluster_type
            )
        }
    
    def set_Z_full(self, Z_full):
        """
        Set or update the hierarchical clustering data.
        
        Args:
            Z_full: The hierarchical clustering data
        """
        self.Z_full = Z_full
        
        # Update scorer
        self.scorer = Scoring(Z_full, self.class_names)
        
        # Update Z_full in all attack instances
        for attack in self.attacks.values():
            attack.Z_full = Z_full
            attack.scorer = self.scorer
    
    def set_cluster_type(self, cluster_type):
        """
        Set or update the cluster type.
        
        Args:
            cluster_type: Type of clustering to use
        """
        self.cluster_type = cluster_type
        
        # Update cluster_type in all attack instances
        for attack in self.attacks.values():
            attack.cluster_type = cluster_type
    
    def set_dataset_type(self, dataset_type):
        """
        Set or update the dataset type.
        
        Args:
            dataset_type: Type of dataset to use
        """
        # If dataset type changes, we may need to reload the model
        if dataset_type != self.dataset_type:
            self.dataset_type = dataset_type
            self.model = None  # Reset model so it will be reloaded with new dataset
            self.load_model()
    
    def perturb_images(self, images, attack_type, **kwargs):
        """
        Generate adversarial examples using the specified attack.
        
        Args:
            images: Input images to perturb
            attack_type: Type of attack to use
            **kwargs: Attack-specific parameters
            
        Returns:
            Adversarial examples
        """
        # Ensure model is loaded
        self.load_model()
        
        if isinstance(attack_type, str):
            attack_type = AttackType[attack_type]
            
        if attack_type not in self.attacks:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Update perturbed_images attribute and return
        self.perturbed_images = self.attacks[attack_type].perturb_images(images, **kwargs)
        return self.perturbed_images
    
    def analyze_attack(self, image_index, original_images, attack_type, **kwargs):
        """
        Run the specified attack and analyze results with adversarial score.
        
        Parameters:
        - image_index: Index of the image to analyze
        - original_images: Array of original images
        - attack_type: Type of attack to use
        - **kwargs: Attack-specific parameters
        
        Returns:
        - Dictionary with analysis results
        """
        # Ensure model is loaded
        self.load_model()
        
        if isinstance(attack_type, str):
            attack_type = AttackType[attack_type]
            
        if attack_type not in self.attacks:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Get the appropriate attack instance
        attack = self.attacks[attack_type]
        
        # Get the threshold for this attack
        threshold = kwargs.pop('threshold', self.get_threshold(attack_type))
        
        # Add threshold to kwargs
        kwargs['threshold'] = threshold
        
        # Run the attack analysis
        return attack.analyze_attack(image_index, original_images, self.class_names, **kwargs)
    
    def run_evaluation(self, original_images, attack_type, num_samples=100, **kwargs):
        """
        Run an attack evaluation on multiple samples.
        
        Parameters:
        - original_images: Array of original images
        - attack_type: Type of attack to use
        - num_samples: Number of samples to evaluate
        - **kwargs: Attack-specific parameters
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Ensure model is loaded
        self.load_model()
        
        if isinstance(attack_type, str):
            attack_type = AttackType[attack_type]
            
        if attack_type not in self.attacks:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # Get the appropriate attack instance
        attack = self.attacks[attack_type]
        
        # Get the threshold for this attack
        threshold = kwargs.pop('threshold', self.get_threshold(attack_type))
        
        # Add threshold to kwargs
        kwargs['threshold'] = threshold
        
        # Run the evaluation
        return attack.run_evaluation(
            original_images[:num_samples], 
            None,  # No true labels needed for detection evaluation
            self.class_names,
            num_samples=num_samples,
            **kwargs
        )
    
    def vulnerability_check(self, images, attack_type=None, num_samples=10, **kwargs):
        """
        Check model vulnerability to specific or all attack types.
        
        Args:
            images: Input images to test
            attack_type: Specific attack type to test (tests all if None)
            num_samples: Number of samples to test
            **kwargs: Attack-specific parameters
            
        Returns:
            Dictionary with vulnerability metrics for each attack type
        """
        # Ensure model is loaded
        self.load_model()
        
        # Limit samples
        if len(images) > num_samples:
            images = images[:num_samples]
        
        results = {}
        
        # If specific attack type requested
        if attack_type is not None:
            if isinstance(attack_type, str):
                attack_type = AttackType[attack_type]
                
            if attack_type not in self.attacks:
                raise ValueError(f"Unsupported attack type: {attack_type}")
                
            # Get threshold for this attack
            threshold = kwargs.pop('threshold', self.get_threshold(attack_type))
            
            # Run vulnerability check with this attack
            return self.attacks[attack_type].vulnerability_check(images, threshold=threshold, **kwargs)
        
        # Otherwise test all attack types
        for attack_type, attack in self.attacks.items():
            # Get appropriate threshold for this attack
            threshold = self.get_threshold(attack_type)
            
            # Set attack-specific parameters
            attack_kwargs = kwargs.copy()
            if attack_type == AttackType.FGSM:
                attack_kwargs.setdefault('epsilon', 0.1)
            elif attack_type == AttackType.PGD:
                attack_kwargs.setdefault('epsilon', 0.1)
                attack_kwargs.setdefault('alpha', 0.01)
                attack_kwargs.setdefault('num_steps', 40)
            elif attack_type == AttackType.DEEPFOOL:
                attack_kwargs.setdefault('num_classes', 10)
                attack_kwargs.setdefault('max_iter', 20)
            
            # Add threshold
            attack_kwargs['threshold'] = threshold
            
            # Run check
            results[attack_type.name] = attack.vulnerability_check(images, **attack_kwargs)
        
        return results
    
    def consistency_check(self, images, attack_type=None, num_samples=5, **kwargs):
        """
        Check consistency of model predictions under different attack parameters.
        
        Args:
            images: Input images to test
            attack_type: Specific attack type to test (tests all if None)
            num_samples: Number of samples to test
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with consistency metrics for each attack type
        """
        # Ensure model is loaded
        self.load_model()
        
        # Limit samples
        if len(images) > num_samples:
            images = images[:num_samples]
        
        results = {}
        
        # If specific attack type requested
        if attack_type is not None:
            if isinstance(attack_type, str):
                attack_type = AttackType[attack_type]
                
            if attack_type not in self.attacks:
                raise ValueError(f"Unsupported attack type: {attack_type}")
                
            # Run consistency check with this attack
            return self.attacks[attack_type].consistency_check(self.model, images, **kwargs)
        
        # Otherwise test all attack types
        for attack_type, attack in self.attacks.items():
            results[attack_type.name] = attack.consistency_check(self.model, images, **kwargs)
        
        return results
    
    def get_scoring(self):
        """
        Get the scoring instance for direct access to scoring functions.
        
        Returns:
            The Scoring instance
        """
        return self.scorer
    
    def calculate_adversarial_score(self, predictions, top_k=5):
        """
        Calculate adversarial score using the scoring module.
        
        Args:
            predictions: Model predictions
            top_k: Number of top predictions to consider
            
        Returns:
            Dictionary with score and detection information
        """
        return self.scorer.calculate_adversarial_score(predictions, top_k)
    
    def is_adversarial(self, predictions, attack_type=None, threshold=None, top_k=5):
        """
        Quick check if predictions are adversarial.
        
        Args:
            predictions: Model predictions
            attack_type: Type of attack to use for threshold selection (if None, uses custom threshold)
            threshold: Custom threshold (if None and attack_type provided, uses default for that attack)
            top_k: Number of top predictions to consider
            
        Returns:
            Boolean indicating if prediction is adversarial
        """
        # If threshold is explicitly provided, use it
        if threshold is not None:
            return self.scorer.is_adversarial(predictions, threshold, top_k)
        
        # If attack type is provided, use its default threshold
        if attack_type is not None:
            if isinstance(attack_type, str):
                attack_type = AttackType[attack_type]
            threshold = self.get_threshold(attack_type)
            return self.scorer.is_adversarial(predictions, threshold, top_k)
        
        # Otherwise use a generic threshold
        return self.scorer.is_adversarial(predictions, 120, top_k)
    
    def predict_with_model(self, images, ensure_batch=True):
        """
        Make predictions with the model, handling different model types.
        
        Args:
            images: Input images
            ensure_batch: Whether to ensure images have batch dimension
            
        Returns:
            Predictions
        """
        # Ensure model is loaded
        self.load_model()
        
        # Ensure images have batch dimension if needed
        if ensure_batch and len(images.shape) == 3:
            import numpy as np
            images = np.expand_dims(images, axis=0)
        
        # Preprocess images
        processed_images = self.preprocess_input(images)
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            # Using model.predict method
            preds = self.model.predict(processed_images, verbose=0)
        else:
            # Using model.__call__ method
            import tensorflow as tf
            preds = self.model(processed_images, training=False)
            if isinstance(preds, tf.Tensor):
                preds = preds.numpy()
        
        return preds
    
    def load_test_images(self, test_folder, labels_dict, num_images=100):
        """
        Load test images from the specified folder.
        
        Args:
            test_folder: Path to the test images
            labels_dict: Dictionary mapping folder names to class labels
            num_images: Maximum number of images to load
            
        Returns:
            List of (image path, label) tuples and loaded image arrays
        """
        import os
        import tensorflow as tf
        import numpy as np
        
        test_metadata = []
        og_x_test = []
        og_y_test = []
        count = 0

        # Get class name mapping if needed
        class_names = list(set(labels_dict.values()))  # unique label names
        synset_to_index = {k: class_names.index(v) for k, v in labels_dict.items()}  # synset ID → index

        for class_folder in os.listdir(test_folder):
            class_path = os.path.join(test_folder, class_folder)

            if not os.path.isdir(class_path):
                continue

            if class_folder not in synset_to_index:
                continue

            class_label = synset_to_index[class_folder]

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)

                if not os.path.isfile(img_path) or not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Add metadata
                test_metadata.append((img_path, class_label))
                
                # Load the image
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                og_x_test.append(img_array)
                og_y_test.append(class_label)

                count += 1
                if count >= num_images:
                    break

            if count >= num_samples:
                break

        og_x_test = np.array(og_x_test)
        og_y_test = np.array(og_y_test)
        
        return test_metadata, og_x_test, og_y_test
    
    def evaluate_detection_performance(self, num_samples=50, thresholds=None):
        """
        Comprehensive evaluation of attack detection performance.
        
        Args:
            num_samples: Number of samples to evaluate
            thresholds: Dictionary of detection thresholds for each attack
            
        Returns:
            Dictionary with evaluation metrics
        """
        import numpy as np
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        import matplotlib.pyplot as plt
        
        if thresholds is None:
            thresholds = {
                'FGSM': self.get_threshold(AttackType.FGSM),
                'PGD': self.get_threshold(AttackType.PGD),
                'DEEPFOOL': self.get_threshold(AttackType.DEEPFOOL)
            }
        
        results = {}
        
        # Ensure we have test images
        if self.og_images is None or len(self.og_images) < num_samples:
            raise ValueError("Original images must be set and contain at least num_samples images")
            
        # Ensure model is loaded
        self.load_model()
        
        original_images = self.og_images[:num_samples]
        
        # Get original predictions and scores for clean images
        clean_scores = []
        clean_preds = []
        
        for i in range(min(num_samples, len(original_images))):
            original_image = original_images[i].astype(np.float32)
            
            # Get original score
            predictions = self.predict_with_model(original_image)
            
            score = self.calculate_adversarial_score(predictions[0])
            clean_scores.append(score['score'])
            clean_preds.append(np.argmax(predictions[0]))
        
        # Evaluate each attack type
        for attack_type in [AttackType.FGSM, AttackType.PGD, AttackType.DEEPFOOL]:
            attack_name = attack_type.name
            attack_results = {
                'scores': [],
                'success_rate': 0,
                'detection_rate': 0,
                'false_positive_rate': 0
            }
            
            successful_attacks = 0
            detected_attacks = 0
            
            # Generate adversarial examples
            for i in range(min(num_samples, len(original_images))):
                original_image = original_images[i].astype(np.float32)
                
                # Get original prediction
                orig_preds = self.predict_with_model(original_image)
                orig_class = np.argmax(orig_preds[0])
                
                # Generate adversarial example
                if attack_type == AttackType.FGSM:
                    adv_image = self.perturb_images(original_image/255.0, attack_type, epsilon=0.1)
                elif attack_type == AttackType.PGD:
                    adv_image = self.perturb_images(original_image/255.0, attack_type, 
                                                epsilon=0.1, alpha=0.01, num_steps=40)
                else:  # DEEPFOOL
                    adv_image = self.perturb_images(original_image/255.0, attack_type, 
                                                num_classes=10, max_iter=20)
                
                # Get adversarial prediction
                adv_preds = self.predict_with_model(adv_image*255.0)
                adv_class = np.argmax(adv_preds[0])
                
                # Check if attack was successful
                if adv_class != orig_class:
                    successful_attacks += 1
                
                # Calculate score
                score = self.calculate_adversarial_score(adv_preds[0])
                attack_results['scores'].append(score['score'])
                
                # Check if detected as adversarial
                if score['score'] > thresholds[attack_name]:
                    detected_attacks += 1
            
            # Calculate metrics
            attack_results['success_rate'] = successful_attacks / num_samples
            attack_results['detection_rate'] = detected_attacks / num_samples
            
            # False positive rate (from clean images)
            false_positives = sum(1 for score in clean_scores if score > thresholds[attack_name])
            attack_results['false_positive_rate'] = false_positives / len(clean_scores)
            
            # Calculate ROC and PR curves
            y_true = [0] * len(clean_scores) + [1] * len(attack_results['scores'])
            y_scores = clean_scores + attack_results['scores']
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            attack_results['roc_auc'] = roc_auc
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            attack_results['pr_auc'] = pr_auc
            
            # Plot ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {attack_name} Detection')
            plt.legend(loc="lower right")
            plt.savefig(f'{attack_name}_roc_curve.png')
            plt.close()
            
            # Store results
            results[attack_name] = attack_results
        
        return results
    
    def visualize_attack_comparison(self, image_indices=None, num_samples=5, epsilon=0.1):
        """
        Create a comprehensive visualization comparing all three attack types on the same images.
        
        Args:
            image_indices: Specific indices to visualize (uses range(num_samples) if None)
            num_samples: Number of images to visualize if image_indices is None
            epsilon: Epsilon value for FGSM and PGD attacks
            
        Returns:
            None (saves visualization to disk)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Ensure model is loaded
        self.load_model()
        
        # Ensure we have images
        if self.og_images is None:
            raise ValueError("Original images must be set before visualization")
        
        # Determine which images to visualize
        if image_indices is None:
            image_indices = range(min(num_samples, len(self.og_images)))
        
        for i in image_indices:
            if i >= len(self.og_images):
                print(f"Warning: Image index {i} out of bounds, skipping")
                continue
                
            original_image = self.og_images[i].astype(np.float32)
            
            # Get original prediction
            orig_preds = self.predict_with_model(original_image)
            orig_class = np.argmax(orig_preds[0])
            
            # Generate adversarial examples for each attack
            fgsm_image = self.perturb_images(original_image/255.0, AttackType.FGSM, epsilon=epsilon)
            pgd_image = self.perturb_images(original_image/255.0, AttackType.PGD, 
                                        epsilon=epsilon, alpha=epsilon/10, num_steps=40)
            deepfool_image = self.perturb_images(original_image/255.0, AttackType.DEEPFOOL, 
                                            num_classes=10, max_iter=20)
            
            # Get predictions
            fgsm_preds = self.predict_with_model(fgsm_image*255.0)
            pgd_preds = self.predict_with_model(pgd_image*255.0)
            deepfool_preds = self.predict_with_model(deepfool_image*255.0)
            
            fgsm_class = np.argmax(fgsm_preds[0])
            pgd_class = np.argmax(pgd_preds[0])
            deepfool_class = np.argmax(deepfool_preds[0])
            
            # Calculate adversarial scores
            orig_score = self.calculate_adversarial_score(orig_preds[0])
            fgsm_score = self.calculate_adversarial_score(fgsm_preds[0])
            pgd_score = self.calculate_adversarial_score(pgd_preds[0])
            deepfool_score = self.calculate_adversarial_score(deepfool_preds[0])
            
            # Compute perturbations (for visualization)
            fgsm_pert = np.abs(fgsm_image*255.0 - original_image)
            pgd_pert = np.abs(pgd_image*255.0 - original_image)
            deepfool_pert = np.abs(deepfool_image*255.0 - original_image)
            
            # Scale perturbations for better visibility
            fgsm_pert = np.clip(fgsm_pert * 10, 0, 255).astype(np.uint8)
            pgd_pert = np.clip(pgd_pert * 10, 0, 255).astype(np.uint8)
            deepfool_pert = np.clip(deepfool_pert * 10, 0, 255).astype(np.uint8)
            
            # Create visualization
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            # Original image
            axes[0, 0].imshow(original_image.astype(np.uint8))
            axes[0, 0].set_title(f"Original\nClass: {self.class_names[orig_class] if orig_class < len(self.class_names) else 'Unknown'}")
            axes[0, 0].axis('off')
            
            # Original score
            axes[0, 1].axis('off')
            axes[0, 1].text(0.5, 0.5, f"Score: {orig_score['score']:.2f}\nAdversarial: {orig_score['is_adversarial']}",
                        horizontalalignment='center', verticalalignment='center')
            
            # Original perturbation (empty)
            axes[0, 2].axis('off')
            
            # FGSM attack
            axes[1, 0].imshow((fgsm_image*255.0).astype(np.uint8))
            axes[1, 0].set_title(f"FGSM Attack\nClass: {self.class_names[fgsm_class] if fgsm_class < len(self.class_names) else 'Unknown'}")
            axes[1, 0].axis('off')
            
            # FGSM score
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, f"Score: {fgsm_score['score']:.2f}\nAdversarial: {fgsm_score['is_adversarial']}",
                        horizontalalignment='center', verticalalignment='center')
            
            # FGSM perturbation
            axes[1, 2].imshow(fgsm_pert)
            axes[1, 2].set_title("FGSM Perturbation (x10)")
            axes[1, 2].axis('off')
            
            # PGD attack
            axes[2, 0].imshow((pgd_image*255.0).astype(np.uint8))
            axes[2, 0].set_title(f"PGD Attack\nClass: {self.class_names[pgd_class] if pgd_class < len(self.class_names) else 'Unknown'}")
            axes[2, 0].axis('off')
            
            # PGD score
            axes[2, 1].axis('off')
            axes[2, 1].text(0.5, 0.5, f"Score: {pgd_score['score']:.2f}\nAdversarial: {pgd_score['is_adversarial']}",
                        horizontalalignment='center', verticalalignment='center')
            
            # PGD perturbation
            axes[2, 2].imshow(pgd_pert)
            axes[2, 2].set_title("PGD Perturbation (x10)")
            axes[2, 2].axis('off')
            
            # DeepFool attack
            axes[3, 0].imshow((deepfool_image*255.0).astype(np.uint8))
            axes[3, 0].set_title(f"DeepFool Attack\nClass: {self.class_names[deepfool_class] if deepfool_class < len(self.class_names) else 'Unknown'}")
            axes[3, 0].axis('off')
            
            # DeepFool score
            axes[3, 1].axis('off')
            axes[3, 1].text(0.5, 0.5, f"Score: {deepfool_score['score']:.2f}\nAdversarial: {deepfool_score['is_adversarial']}",
                        horizontalalignment='center', verticalalignment='center')
            
            # DeepFool perturbation
            axes[3, 2].imshow(deepfool_pert)
            axes[3, 2].set_title("DeepFool Perturbation (x10)")
            axes[3, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"attack_comparison_image_{i}.png")
            plt.close()
            
            print(f"Saved comparison for image {i}")