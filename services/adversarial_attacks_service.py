from utilss.classes.adversarial_dataset import AdversarialDataset
from utilss.classes.adversarial_detector import AdversarialDetector
from utilss.classes.score_calculator import ScoreCalculator
from utilss.photos_utils import preprocess_image, encode_image_to_base64, deprocess_resnet_image, preprocess_deepfool_image, preprocess_loaded_image
from services.dataset_service import get_dataset_labels
from utilss.classes.adversarial_attacks.adversarial_attack_factory import get_attack
from services.models_service import get_top_k_predictions, query_model, get_user_models_info, get_model_specific_file
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import os
import logging
from utilss.s3_utils import get_users_s3_client
from typing import Optional, Tuple
logger = logging.getLogger(__name__)


def _parse_s3_path(model_file: Optional[str], s3_bucket: str) -> Tuple[str, str]:
    """
    Parse the model file path to get the S3 bucket and key.
    
    Args:
        model_file (Optional[str]): The path to the model file, either an S3 path or relative path
        s3_bucket (str): The default S3 bucket name to use if model_file is not an S3 path
        
    Returns:
        Tuple[str, str]: A tuple containing (bucket, key)
        
    Raises:
        ValueError: If model_file is None
    """
    if model_file is None:
        raise ValueError("model_file is None. Cannot proceed to load model.")
        
    if model_file.startswith('s3://'):
        parts = model_file.replace('s3://', '').split('/', 1)
        return parts[0], parts[1]
    else:
        return s3_bucket, model_file


def _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_filename, dataset) -> dict:
    logger.info("Creating adversarial dataset")
    adversarial_dataset = AdversarialDataset(Z_file, clean_images, adversarial_images, model_filename, dataset)
    X_train, y_train, X_test, y_test = adversarial_dataset.create_logistic_regression_dataset()
    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }   
    return result



def create_logistic_regression_detector(model_id, graph_type, clean_images, adversarial_images, user):
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_graph_folder = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "graph_folder")
        model_file = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "model_file")
        Z_file = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "Z_file")
        
    adversarial_detector = AdversarialDetector(model_graph_folder)
    adversarial_dataset = _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_file, model_info["dataset"])
    adversarial_detector.train_adversarial_detector(adversarial_dataset)

    return adversarial_detector



def detect_adversarial_image(model_id, graph_type, image, user, detector_filename):
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
    logger.info("Detecting adversarial image")
    logger.debug(f"Model ID: {model_id}, Graph Type: {graph_type}")

    # Initialize S3 client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

    model_info = get_user_models_info(user, model_id)

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_graph_folder = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "graph_folder")
        model_file = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "model_file")

        dataset = model_info["dataset"]
        Z_full = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "Z_file")
        labels = get_dataset_labels(dataset)

        # Check if model file exists in S3
        try:
            bucket, key = _parse_s3_path(model_file, s3_bucket)
            
            # Check if the file exists
            s3_client.head_object(Bucket=bucket, Key=key)
            
            # Use temporary directory to load model
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = os.path.join(temp_dir, 'model.keras')
                s3_client.download_file(bucket, key, temp_model_path)
                model = tf.keras.models.load_model(temp_model_path)
                logger.info(f"Model loaded successfully from S3 '{bucket}/{key}'.")
                
        except Exception as e:
            raise ValueError(f"Failed to load model file from S3: {str(e)}")
        
    image_preprocessed = preprocess_loaded_image(model, image)

    detector = AdversarialDetector(model_graph_folder, detector_filename)
    score_calculator = ScoreCalculator(Z_full, labels)
    
    # Get predictions from the original model
    preds = model.predict(image_preprocessed, verbose=0)
    
    # Calculate the adversarial score (or other features)
    score = score_calculator.calculate_adversarial_score(preds[0])
    
    # Use the detector to classify the image
    feature = [[score]]  # Wrap the score in a 2D array
    label = detector.predict(feature)[0]  # Predict the label (0 = clean, 1 = adversarial)
    proba = detector.predict_proba(feature)[0][1]  # Probability of being adversarial
    detection_result = 'Adversarial' if label == 1 else 'Clean'

    logger.info(f"Adversarial score: {score}, Label: {label}")

    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    image_array = np.array(pil_image)
    image_base64 = encode_image_to_base64(image_array)

    image_preprocessed = preprocess_loaded_image(model, image)
    image_predictions = get_top_k_predictions(model, image_preprocessed, labels)
    
    return {
        "image": image_base64,
        "predictions": image_predictions,
        "result": detection_result,
        "probability": proba
        }

def analysis_adversarial_image(model_id, graph_type, attack_type, image, user, detector_filename, **kwargs):
    logger.info("Analyzing adversarial image")
    logger.debug(f"Model ID: {model_id}, Graph Type: {graph_type}, Attack Type: {attack_type}")

    # Initialize S3 client
    s3_client = get_users_s3_client()
    s3_bucket = os.getenv("S3_USERS_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")

    model_info = get_user_models_info(user, model_id)

    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_graph_folder = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "graph_folder")
        model_file = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "model_file")
        Z_full = get_model_specific_file(user.get_user_folder(), model_info, graph_type, "Z_file")
        
        # Check if model file exists in S3
        try:
            bucket, key = _parse_s3_path(model_file, s3_bucket)
            
            # Check if the file exists
            s3_client.head_object(Bucket=bucket, Key=key)
            
            # Use temporary directory to load model
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_model_path = os.path.join(temp_dir, 'model.keras')
                s3_client.download_file(bucket, key, temp_model_path)
                model = tf.keras.models.load_model(temp_model_path)
                logger.info(f"Model loaded successfully from S3 '{bucket}/{key}'.")
                
        except Exception as e:
            raise ValueError(f"Failed to load model file from S3: {str(e)}")
        
        dataset = model_info["dataset"]
        labels = get_dataset_labels(dataset)
        detector = AdversarialDetector(model_graph_folder, detector_filename)
        score_calculator = ScoreCalculator(Z_full, labels)

        if attack_type == "deepfool":
            preprocessed_image = preprocess_deepfool_image(model, image)
        else:
            preprocessed_image = preprocess_loaded_image(model, image)

        adversarial_attack = get_attack(attack_type, class_names=labels, **kwargs)
        adversarial_image = adversarial_attack.attack(model, preprocessed_image)

        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        original_size = pil_image.size
        image_array = np.array(pil_image)
        original_image_base64 = encode_image_to_base64(image_array)
        if attack_type != "deepfool":
            adversarial_image = deprocess_resnet_image(adversarial_image)
        else:
            # DeepFool output is [0,1] float, scale to [0,255]
            adversarial_image = np.clip(adversarial_image * 255.0, 0, 255).astype(np.uint8)
        # Convert adversarial_image (numpy array) to PIL Image
        adv_pil = Image.fromarray(adversarial_image.astype(np.uint8))

        # Resize back to original size
        adv_resized = adv_pil.resize(original_size, Image.Resampling.BILINEAR)

        # If you need it as a numpy array again:
        adv_resized_np = np.array(adv_resized)
        adversarial_image_base64 = encode_image_to_base64(adv_resized_np)

        original_image_preprocessed = preprocess_loaded_image(model, image)
        adversarial_image_preprocessed = preprocess_image(model, adversarial_image)

        # Get top K predictions for the original image
        original_predictions = get_top_k_predictions(model, original_image_preprocessed, labels)
        original_verbal_explaination = query_model(original_predictions[0][0], model_id, graph_type, user)
        orignal_pred = model.predict(original_image_preprocessed, verbose=0)
        original_score = score_calculator.calculate_adversarial_score(orignal_pred[0])
        original_feature = [[original_score]]  # Wrap the score in a 2D array
        original_label = detector.predict(original_feature)[0]  # Predict the label (0 = clean, 1 = adversarial)
        original_proba = detector.predict_proba(original_feature)[0][1]  # Probability of being adversarial
        original_detection_result = 'Adversarial' if original_label == 1 else 'Clean'

        # Get top K predictions for the adversarial image
        adversarial_predictions = get_top_k_predictions(model, adversarial_image_preprocessed, labels)
        adversarial_verbal_explaination = query_model(adversarial_predictions[0][0], model_id, graph_type, user)
        adversarial_pred = model.predict(adversarial_image_preprocessed, verbose=0)
        adversarial_score = score_calculator.calculate_adversarial_score(adversarial_pred[0])
        adversarial_feature = [[adversarial_score]]  # Wrap the score in a 2D array
        adversarial_label = detector.predict(adversarial_feature)[0]
        adversarial_proba = detector.predict_proba(adversarial_feature)[0][1]
        adversarial_detection_result = 'Adversarial' if adversarial_label == 1 else 'Clean'
        logger.info(f"Original Image - Score: {original_score}, Label: {original_label}, Probability: {original_proba}")
        logger.info(f"Adversarial Image - Score: {adversarial_score}, Label: {adversarial_label}, Probability: {adversarial_proba}")

        # Return both images as Base64 strings
        return {
            "original_image": original_image_base64,
            "original_predictions": original_predictions,
            "original_verbal_explaination": original_verbal_explaination,
            "original_probability": original_proba,
            "original_detection_result": original_detection_result,
            "adversarial_image": adversarial_image_base64,
            "adversarial_predictions": adversarial_predictions,
            "adversarial_verbal_explaination": adversarial_verbal_explaination,
            "adversarial_probability": adversarial_proba,
            "adversarial_detection_result": adversarial_detection_result
        }
    

def get_detector_list(user, model_id, graph_type):
    """
    Get the list of adversarial detectors for the user of the specific model graph type.
    
    Parameters:
    - user: The user object
    
    Returns:
    - List of detectors names
    """
    logger.info("Getting detector list")
    model_info = get_user_models_info(user, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    
    S3_BUCKET = os.getenv("S3_USERS_BUCKET_NAME")
    if not S3_BUCKET:
        raise ValueError("S3_USERS_BUCKET_NAME environment variable is required")
    
    user_folder = user.get_user_folder()
    model_subfolder = f"{user_folder}/{model_info['model_id']}"

    model_graph_folder = f"{model_subfolder}/{graph_type}"
    detector_prefix = f"{model_graph_folder}/logistic_regression_model"
    
    s3_client = get_users_s3_client()
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=S3_BUCKET, Prefix=detector_prefix)

    detector_keys = []
    for page in page_iterator:
        for obj in page.get('Contents', []):
            detector_keys.append(os.path.basename(obj['Key']))

    return detector_keys
