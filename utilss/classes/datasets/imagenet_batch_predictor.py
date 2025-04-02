import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ImageNetBatchPredictor:
    def __init__(self, model, batch_size=64, img_size=(224, 224), num_workers=16):
        self.model = model
        self.batch_size = batch_size

    def predict_batch(self, image_paths, top_k=5):
        """Process multiple images at once in a batch"""
        batch_size = len(image_paths)
        img_arrays = np.zeros((batch_size, 224, 224, 3))
        
        for i, path in enumerate(image_paths):
            img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_arrays[i] = img_array
        
        batch_predictions = self.model.predict(img_arrays)
        return [decode_predictions(np.expand_dims(pred, axis=0), top=top_k)[0] for pred in batch_predictions]