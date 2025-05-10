from utilss.classes.adversarial_dataset import AdversarialDataset
from utilss.classes.adversarial_detector import AdversarialDetector
from utilss.classes.score_calculator import ScoreCalculator
from utilss.files_utils import get_user_models_info, get_model_files, get_labels_from_dataset_info
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os
import json


def _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_filename, dataset) -> AdversarialDataset:
    """Create an adversarial dataset based on the provided configuration."""
    adversarial_dataset = AdversarialDataset(Z_file, clean_images, adversarial_images, model_filename, dataset)
    X_train, y_train, X_test, y_test = adversarial_dataset.create_logistic_regression_dataset()
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

def create_logistic_regression_detector(model_id: str, graph_type: str, clean_images: list, adversarial_images: list, user_folder: str):
    model_info = get_user_models_info(user_folder, model_id)

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user_folder, model_info, graph_type)
        model_graph_folder = model_files["model_graph_folder"]
        model_file = model_files["model_file"]
        Z_file = model_files["Z_file"]
        
    adversarial_dataset = _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_file, model_info["dataset"])
    adversarial_detector = AdversarialDetector(model_graph_folder)
    adversarial_detector.train_adversarial_detector(adversarial_dataset)

    return adversarial_dataset

def detect_adversarial_image(model_id, graph_type, image, user_folder):
    """
    Detect if an image is adversarial using the trained logistic regression detector.
    
    Parameters:
    - model: The original model being attacked
    - detector: The trained logistic regression detector
    - Z_full: Hierarchical clustering data
    - class_names: List of class names
    - image: The input image to check
    
    Returns:
    - Prediction (Clean or Adversarial) and probability
    """

    model_info = get_user_models_info(user_folder, model_id)

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user_folder, model_info, graph_type)
        model_graph_folder = model_files["model_graph_folder"]
        model_file = model_files["model_file"]

        dataset = model_info["dataset"]
        Z_full = model_files["Z_file"]
        labels = get_labels_from_dataset_info(dataset)

        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
            print(f"Model loaded successfully from '{model_file}'.")
        else:
            raise ValueError(f"Model file {model_file} does not exist")
        
        expected_shape = model.input_shape
        input_height, input_width = expected_shape[1], expected_shape[2]
        print(f"Model expects input shape: {expected_shape}")

        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Resize the image based on the model's expected input size
        pil_image = pil_image.resize((input_width, input_height))
        
        # Convert the image to a NumPy array and normalize pixel values to [0, 1]
        image_array = np.array(pil_image) / 255.0
        
        print(f"Resized image array shape: {image_array.shape}") 
        
        # Add a batch dimension (models expect input shape: [batch_size, height, width, channels])
        image_preprocessed = np.expand_dims(image_array, axis=0)
        
        print(f"Preprocessed image shape before prediction: {image_preprocessed.shape}")

        detector = AdversarialDetector(model_graph_folder)

        score_calculator = ScoreCalculator(Z_full, labels)
    
    # Get predictions from the original model
    preds = model.predict(image_preprocessed, verbose=0)
    
    # Calculate the adversarial score (or other features)
    score = score_calculator.calculate_adversarial_score(preds[0])
    
    # Use the detector to classify the image
    feature = [[score]]  # Wrap the score in a 2D array
    label = detector.predict(feature)[0]  # Predict the label (0 = clean, 1 = adversarial)
    
    return ('Adversarial' if label == 1 else 'Clean')


    

    

    


    

