import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np

class AdversarialDetector:
    def __init__(self, model_folder, threshold=0.5):
        self.model_folder = model_folder
        detector_filename =  f'{model_folder}/logistic_regression_model.pkl'
        if os.path.exists(detector_filename):
            self.detector, self.threshold = joblib.load(detector_filename)
            print(f"Detector model loaded from '{detector_filename}'.")
        else:
            self.detector = None
        self.threshold = threshold

    def does_detector_exist(self):
        """
        Check if the detector model exists.
        
        Returns:
        - bool: True if the detector model exists, False otherwise.
        """
        return self.detector is not None

    def predict(self, X):
        # Predict probabilities
        if self.detector is None:
            raise ValueError("Detector model is not trained or loaded.")
        if X is None or len(X) == 0:
            raise ValueError("Input data is empty or None.")
        
        y_pred_proba = self.detector.predict_proba(X)[:, 1]
        # Apply the custom threshold
        return (y_pred_proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        if self.detector is None:
            raise ValueError("Detector model is not trained or loaded.")
        if X is None or len(X) == 0:
            raise ValueError("Input data is empty or None.")
        # Return predicted probabilities
        return self.detector.predict_proba(X)

    def train_adversarial_detector(self, dataset):
        """
        Train a logistic regression model to detect adversarial examples across different attack types.
        
        Parameters:
        - model: The model being attacked
        - Z_full: Hierarchical clustering data
        - class_names: List of class names
        - num_samples: Number of samples to use for training
        
        Returns:
        - Trained detector model and evaluation metrics
        """
        
        X_train, y_train = dataset['X_train'], dataset['y_train']

        # Train logistic regression model
        print("Training adversarial detector...")
        detector = LogisticRegression(max_iter=1000, class_weight='balanced')
        detector.fit(X_train, y_train)
        self.detector = detector

        X_test, y_test = dataset['X_test'], dataset['y_test']

        # Predict probabilities for the test set
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Find the optimal threshold (closest to top-left corner)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        self.threshold = optimal_threshold

        # Saving the lr-model
        print("Training completed.")
        joblib.dump((detector, optimal_threshold), f'{self.model_folder}/logistic_regression_model.pkl')
        print(f"Detector model saved as f'{self.model_folder}/logistic_regression_model.pkl'")

        
        return detector
    
