from sklearn.model_selection import train_test_split
import numpy as np
from .score_calculator import ScoreCalculator
import tensorflow as tf
import os

class AdversarialDataset:
    def __init__(self, clear_images, adversarial_images, base_directory):
        self.clear_images = clear_images
        self.adversarial_images = adversarial_images
        self.base_directory = base_directory
        # Load images if not provided
        if clear_images is None:
            self.clear_images = self._load_images_from_directory("data/datasets/imagenet/clear")
        else:
            self.clear_images = clear_images

        if adversarial_images is None:
            self.adversarial_images = self._load_images_from_directory("data/datasets/imagenet/adversarial")
        else:
            self.adversarial_images = adversarial_images
        z_file_path = os.path.join(self.base_directory, "matrix_dendrogram.pkl")
        self.score_calculator = ScoreCalculator(z_file_path, class_names=None)

    def _load_images_from_directory(self, directory):
        """
        Load images from a given directory. Assumes images are stored as .npy files.
        """
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                file_path = os.path.join(directory, filename)
                images.append(np.load(file_path))
        return images

    def save_images_to_directories(self):
        """
        Save clear and adversarial images into separate directories.
        """

        # Ensure the directories exist
        os.makedirs(self.clear_dir, exist_ok=True)
        os.makedirs(self.adversarial_dir, exist_ok=True)

        # Save clear images
        for i, img in enumerate(self.clear_images):
            np.save(os.path.join(self.clear_dir, f"clear_{i}.npy"), img.numpy())

        print(f"Saved {len(self.clear_images)} clear images to {self.clear_dir}")

        # Save adversarial images
        for i, img in enumerate(self.adversarial_images):
            np.save(os.path.join(self.adversarial_dir, f"adversarial_{i}.npy"), img.numpy())

        print(f"Saved {len(self.adversarial_images)} adversarial images to {self.adversarial_dir}")

    def load_raw_image(file_path):
        """
        Load a raw adversarial example that was saved as a numpy array
        """
        # Load the numpy array and convert back to tensor
        img_example = np.load(file_path)
        return tf.convert_to_tensor(img_example, dtype=tf.float32)
    
    def calculate_scores(self, model, dir, label):
        scores = []
        for filename in os.listdir(dir):
            if filename.endswith(".npy") and filename.startswith(label):
                file_path = os.path.join(dir, filename)
                loaded_adversarial = self.load_raw_image(file_path)

                score = ScoreCalculator.calculate_adversarial_score(
                    model.predict(loaded_adversarial))
                scores.append(score)
        return scores

    def create_logistic_regression_dataset(self, model, clean_folder, adversarial_folder):
        # Initialize data collection
        scores = []
        labels = []

        try:
            org_scores, org_labels = self.calculate_scores(model, clean_folder, label="clear_")
            scores.extend(org_scores)
            labels.extend(org_labels)
        except Exception as e:
            print(f"Error processing clean image: {e}")
        
        # Generate features for PGD attacks
        print("Generating PGD attack features...")
        try:
            pgd_scores, pgd_labels = self.calculate_scores(model, adversarial_folder, label="adversarial_")
            scores.extend(pgd_scores)
            labels.extend(pgd_labels)
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

        # print(f"X shape: {X.shape}")
        # print(f"y shape: {y.shape}")
        # print(f"X: {X}")
        # print(f"y: {y}")
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # joblib.dump((X_train, y_train, X_test, y_test), f'.{self.base_directory}/{filename}.pkl')
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Clean samples: {sum(y_train == 0)}, Adversarial samples: {sum(y_train == 1)}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Clean samples: {sum(y_test == 0)}, Adversarial samples: {sum(y_test == 1)}")

        return X_train, y_train, X_test, y_test
