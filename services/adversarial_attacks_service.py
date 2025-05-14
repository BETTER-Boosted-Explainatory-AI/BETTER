from utilss.classes.adversarial_dataset import AdversarialDataset
from utilss.classes.adversarial_detector import AdversarialDetector
from utilss.classes.score_calculator import ScoreCalculator
from utilss.files_utils import get_user_models_info, get_model_files, get_labels_from_dataset_info, preprocess_image, encode_image_to_base64, deprocess_resnet_image, preprocess_deepfool_image, preprocess_loaded_image
from utilss.classes.adversarial_attacks.adversarial_attack_factory import get_attack
from services.models_service import get_top_k_predictions
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import os


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
        
    adversarial_detector = AdversarialDetector(model_graph_folder)
    if adversarial_detector.does_detector_exist():
        print("Adversarial detector already exists.")
        return adversarial_detector
    else:
        adversarial_dataset = _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_file, model_info["dataset"])
        adversarial_detector.train_adversarial_detector(adversarial_dataset)

    return adversarial_detector

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
        
    image_preprocessed = preprocess_loaded_image(model, image)

    print(f"Image preprocessed successfully.")

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

def analysis_adversarial_image(model_id, graph_type, attack_type ,image, user_folder, epsilon, alpha, overshoot, num_steps, classes_number):
    model_info = get_user_models_info(user_folder, model_id)

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_files = get_model_files(user_folder, model_info, graph_type)
        model_file = model_files["model_file"]
        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
            print(f"Model loaded successfully from '{model_file}'.")
        else:
            raise ValueError(f"Model file {model_file} does not exist")
        
        dataset = model_info["dataset"]
        labels = get_labels_from_dataset_info(dataset)

        if attack_type == "deepfool":
            preprocessed_image = preprocess_deepfool_image(model, image)
        else:
            preprocessed_image = preprocess_loaded_image(model, image)

        adversarial_attack = get_attack(attack_type, class_names=labels)
        adversarial_image = adversarial_attack.attack(model, preprocessed_image)

        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        image_array = np.array(pil_image)
        original_image_base64 = encode_image_to_base64(image_array)
        if attack_type != "deepfool":
            adversarial_image = deprocess_resnet_image(adversarial_image)
        adversarial_image_base64 = encode_image_to_base64(adversarial_image)

        original_image_preprocessed = preprocess_loaded_image(model, image)
        adversarial_image_preprocessed = preprocess_image(model, adversarial_image)

        # Get top K predictions for the original image
        original_predictions = get_top_k_predictions(model, original_image_preprocessed, labels)

        print(f"Top K predictions for the original image:")
        for k_label, k_prob in original_predictions:
            print(f"{k_label}: {k_prob:.4f}")
        # Get top K predictions for the adversarial image
        adversarial_predictions = get_top_k_predictions(model, adversarial_image_preprocessed, labels)
        print(f"Top K predictions for the adversarial image:")
        for k_label, k_prob in adversarial_predictions:
            print(f"{k_label}: {k_prob:.4f}")
        # Return both images as Base64 strings
        return {
            "original_image": original_image_base64,
            "original_predictions": original_predictions,
            "adversarial_image": adversarial_image_base64,
            "adversarial_predictions": adversarial_predictions,
        }
        




    

    

    


    

