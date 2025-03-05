import os
import pickle
import numpy as np
from .dataset import Dataset
from globalvars import cifar100_labels

class Cifar100(Dataset):
    def __init__(self):
        super().__init__("cifar100",1e-5, 500, cifar100_labels)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load(self, name):
        dataset_path = os.path.join("data", "datasets", name)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from: {dataset_path}")
            self.x_train, self.y_train = self.load_train_batches(dataset_path)
            self.x_test, self.y_test = self.load_batch(os.path.join(dataset_path, 'test_batch'))
            print("Dataset loaded successfully")
        else:
            print(f"Dataset path {dataset_path} does not exist")
    
    def load_train_batches(self, dataset_path):
        x_train = []
        y_train = []
        for i in range(1, 6):
            batch_path = os.path.join(dataset_path, f'data_batch_{i}')
            data, labels = self.load_batch(batch_path)
            x_train.append(data)
            y_train.append(labels)
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        return x_train, y_train
    
    def load_batch(self, batch_path):
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape and transpose to (num_samples, 32, 32, 3)
            labels = np.array(labels)
        return data, labels

    def get_train_image_by_id(self, image_id):
        # Check if the image_id is within the range of training data
        if image_id < len(self.x_train):
            image = self.x_train[image_id]
            label = self.y_train[image_id]
        else:
            raise ValueError("Invalid image_id")

        return image, label
    
    def get_test_image_by_id(self, image_id):
        if image_id < len(self.x_test):
            image = self.x_test[image_id]
            label = self.y_test[image_id]
        else:
            raise ValueError("Invalid image_id")

        return image, label