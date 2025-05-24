import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from data.datasets.cifar100_info import CIFAR100_INFO


class Cifar100BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.buffer_images = []  # To store images
        self.buffer_labels = []  # To store corresponding labels
        self.buffer_results = []  # To store batch results

    def get_top_predictions(self, images, top):
        """
        Process a batch of images and return the top predictions for each image in the batch.
        """
        # Perform prediction on the entire batch
        batch_preds = self.model.predict(np.array(images))

        # Process each image's predictions in the batch
        batch_results = []
        for pred in batch_preds:
            print(f"Prediction: {pred}")
            top_indices = pred.argsort()[-top:][::-1]
            batch_results.append([(i, CIFAR100_INFO["labels"][i], pred[i]) for i in top_indices])
        
        return batch_results    # Returns a list of results for each image in the batch
    