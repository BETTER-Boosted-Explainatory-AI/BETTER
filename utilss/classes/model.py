import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import f1_score
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class Model:
    def __init__(self, model, top_k, percent, model_filename):
        self.model = model
        self.model_filename = model_filename
        self.top_k = top_k
        self.percent = percent
        self.size = (224, 224)
        self.accuracy = -1
        self.loss = -1
        self.f1score = -1

## Need to change the load and save models to aws s3 bucket in the future
    def load_model(self):
        if os.path.exists(self.model_filename):
            self.model = tf.keras.models.load_model(self.model_filename)
            print(f'Model {self.model_filename} has been loaded')
        else:
            print(f'Model {self.model_filename} does not exist')

    def save_model(self):
        self.model.save(self.model_filename)
        print(f'Model has been saved: {self.model_filename}') 
    
    def model_accuracy(self, x_test, y_test):
        if self.accuracy != -1 & self.loss != -1:
            return self.accuracy
        ## cifar100 
        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        batch_size = 64  # Adjust the batch size as needed
        self.loss, self.accuracy = self.model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
        print(f'Model accuracy: {self.accuracy:.4f}')
        print(f'Model loss: {self.loss:.4f}')
    
    def model_accuracy_selected(self, dataset_instance, selected_labels):        
        x_test = dataset_instance.x_test
        y_test = dataset_instance.y_test

        x_test = preprocess_input(x_test)
        y_test = to_categorical(y_test, num_classes=100)

        # validation for specific labels accuracy using mask
        selected_indices = [selected_labels.index(label) for label in selected_labels]

        # Create masks for train and test sets
        test_mask = np.isin(np.argmax(y_test, axis=1), selected_indices)

        # Filter x_test, and y_test
        x_test_filtered = x_test[test_mask]
        y_test_filtered = y_test[test_mask]

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        test_fitlered_loss, test_filtered_accuracy = self.model.evaluate(x_test_filtered, y_test_filtered, verbose=0)
        print(f'selected Labels accuracy: {test_filtered_accuracy:.4f}')
        print(f'selected Labels loss: {test_fitlered_loss:.4f}')

        return test_filtered_accuracy, test_fitlered_loss


    def model_f1score(self, x_test, y_test):
        if self.f1score != -1:
            return self.f1score
        y_test_pred = self.model.predict(x_test)

        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(y_test, axis=1)

        f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        print(f'F1 Score (full dataset): {f1:.4f}')
    
    def model_f1score_selected(self, x_test_filtered, y_test_filtered):
        y_test_filtered_pred = self.model.predict(x_test_filtered)
        y_test_filtered_pred_classes = np.argmax(y_test_filtered_pred, axis=1)
        y_test_filtered_true_classes = np.argmax(y_test_filtered, axis=1)

        f1_filtered = f1_score(y_test_filtered_true_classes, y_test_filtered_pred_classes, average='weighted')
        print(f'F1 Score (selected labels): {f1_filtered:.4f}')

    def predict(self, image_path, top_k=4):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=top_k)

        return decoded_predictions