import os
import pickle
import numpy as np
from .dataset import Dataset
from globalvars import cifar100_labels
import matplotlib.pyplot as plt
from data.datasets.cifar100_info import CIFAR100_INFO

class Cifar100(Dataset):
    def __init__(self):
        super().__init__(CIFAR100_INFO["dataset"], CIFAR100_INFO["threshold"], CIFAR100_INFO["infinity"], CIFAR100_INFO["labels"])
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_train_mapped = None
        self.y_test_mapped = None 

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')  # Load data
        return data_dict

    def load(self, name):
        dataset_path = os.path.join("data", "datasets", name)
        if os.path.exists(dataset_path):
            print(f"Loading dataset from: {dataset_path}")

            train_batch = self.unpickle(os.path.join(dataset_path, "train"))
            print(f"Train batch keys: {train_batch.keys()}")
            print(f"Train batch data shape: {train_batch[b'data'].shape}")

            self.x_train = train_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.y_train = np.array(train_batch[b'fine_labels'])

            test_batch = self.unpickle(os.path.join(dataset_path, "test"))
            print(f"Test batch keys: {test_batch.keys()}")
            print(f"Test batch data shape: {test_batch[b'data'].shape}")

            self.x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.y_test = np.array(test_batch[b'fine_labels'])
            self.y_train_mapped = [self.label_to_class_name(label) for label in self.y_train]
            self.y_test_mapped = [self.label_to_class_name(label) for label in self.y_test]

            print("Dataset loaded successfully")
            return True
        else:
            print(f"Dataset path {dataset_path} does not exist")
            return False
    
    def one_hot_to_class_name_auto(self, one_hot_vector):
        return self.labels[np.argmax(one_hot_vector)] 

    def label_to_class_name(self, label):
        return self.labels[label]

    def get_train_image_by_id(self, image_id):
        # Check if the image_id is within the range of training data
        if image_id < len(self.x_train):
            image = self.x_train[image_id]
            label = self.y_train_mapped[image_id]
            print(f"Train image ID {image_id}: label {label}") 
        else:
            raise ValueError("Invalid image_id")

        return image, label
    
    def get_test_image_by_id(self, image_id):
        if image_id < len(self.x_test):
            image = self.x_test[image_id]
            label = self.y_test_mapped[image_id]
            print(f"Test image ID {image_id}: label {label}")
        else:
            raise ValueError("Invalid image_id")

        return image, label