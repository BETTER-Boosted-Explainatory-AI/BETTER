import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

class Model:
    def __init__(self, model, top_k, percent, model_filename):
        self.model = model
        self.model_filename = model_filename
        self.top_k = top_k
        self.percent = percent
        self.size = (224, 224)
        self.accuracy = -1
        self.f1score = -1

    def load_model(self):
        # need to write
        return None

    def save_model(self):
        # need to write
        return None
    
    def model_accuracy(self):
        if self.accuracy != -1:
            return self.accuracy
        # need to write

    def model_f1score(self):
        if self.f1score != -1:
            return self.f1score
        # need to write

    def predict(self, image_path, top_k=4):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=top_k)

        return decoded_predictions