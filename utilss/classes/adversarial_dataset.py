from sklearn.model_selection import train_test_split
import numpy as np
from utilss.classes.score_calculator import ScoreCalculator
from utilss.files_utils import load_numpy_from_directory
from services.dataset_service import _get_dataset_labels
import tensorflow as tf
import os

class AdversarialDataset:
    def __init__(self, Z_file, clean_images, adversarial_images, model_filename, dataset):
        self.Z_matrix = Z_file
        self.dataset = dataset
        # Load the model
        try:
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"Model file '{model_filename}' not found.")
            
            self.model = tf.keras.models.load_model(model_filename)
            print(f"Model loaded successfully from '{model_filename}'.")
        except Exception as e:
            raise ValueError(f"Error loading model from '{model_filename}': {e}")
        
        DATASET_PATH = os.getenv("DATASETS_PATH")
        if DATASET_PATH is None:
            raise ValueError("DATASET_PATH environment variable is not set.")


        if clean_images is None and adversarial_images is None:
            self.clear_images = load_numpy_from_directory(self.model, f"{DATASET_PATH}{dataset}/clean")
            self.adversarial_images = load_numpy_from_directory(self.model, f"{DATASET_PATH}{dataset}/adversarial")
        else:
            self.clear_images = load_numpy_from_directory(self.model,clean_images)
            self.adversarial_images = load_numpy_from_directory(self.model,adversarial_images)

        self.labels = _get_dataset_labels(dataset)
        if self.labels is None:
            raise ValueError(f"info file for the dataset {dataset} not found'.")      

        self.score_calculator = ScoreCalculator(self.Z_matrix, self.labels)

    def create_logistic_regression_dataset(self):
        scores = []
        labels = []

        print("getting preprocess function...")

        try:
            for image in self.clear_images[:50]:
                score = self.score_calculator.calculate_adversarial_score(self.model.predict(image))
                scores.append(score)
                labels.append(0)
        except Exception as e:
            print(f"Error processing clean image: {e}")
        
        # Generate features for PGD attacks
        print("Generating attack features...")
        try:
            for adv_image in self.adversarial_images[:50]:
                score = self.score_calculator.calculate_adversarial_score(self.model.predict(adv_image))
                scores.append(score)
                labels.append(1)
        except Exception as e:
            print(f"Error processing PGD attack on image: {e}")

        print("labels:", labels)
        print("scores:", scores)
        
        # Convert to numpy arrays
        X = np.array(scores)
        y = np.array(labels)

        # Reshape X to ensure it is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Clean samples: {sum(y_train == 0)}, Adversarial samples: {sum(y_train == 1)}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Clean samples: {sum(y_test == 0)}, Adversarial samples: {sum(y_test == 1)}")

        return X_train, y_train, X_test, y_test
