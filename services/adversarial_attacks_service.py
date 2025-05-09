from utilss.classes.adversarial_dataset import AdversarialDataset
from utilss.classes.adversarial_detector import AdversarialDetector
from utilss.files_utils import get_model_info
import os
import json


def _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_filename, dataset) -> AdversarialDataset:
    """Create an adversarial dataset based on the provided configuration."""
    adversarial_dataset = AdversarialDataset(Z_file, clean_images, adversarial_images, model_filename, dataset)
    X_train, y_train, X_test, y_test = adversarial_dataset.create_logistic_regression_dataset()
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

def create_logistic_regression_detector(model_id: str, graph_type: str, clean_images: list, adversarial_images: list, user_folder: str):
    models_json_path = os.path.join(user_folder, "models.json")
    if os.path.exists(models_json_path):
        with open(models_json_path, "r") as json_file:
            models_data = json.load(json_file)
    else:
        models_data = []
    
    model_info = get_model_info(models_data, model_id)
    if model_info is None:
        raise ValueError(f"Model ID {model_id} not found in models.json")
    else:
        model_subfolder = os.path.join(user_folder, model_info["model_id"])
        model_file = os.path.join(model_subfolder, model_info["file_name"])
        if not os.path.exists(model_file):
            raise ValueError(f"Model file {model_file} does not exist")
        model_graph_folder = os.path.join(model_subfolder, graph_type)
        Z_file = os.path.join(model_graph_folder, f"{graph_type}_dendrogram.pkl")
        if not os.path.exists(Z_file):
            raise ValueError(f"Z file {Z_file} does not exist")
        
    adversarial_dataset = _create_adversarial_dataset(Z_file, clean_images, adversarial_images, model_file, model_info["dataset"])
    adversarial_detector = AdversarialDetector(model_graph_folder)
    adversarial_detector.train_adversarial_detector(adversarial_dataset)

    return adversarial_dataset


    

    

    


    

