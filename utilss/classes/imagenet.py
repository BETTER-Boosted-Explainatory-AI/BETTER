import os
import time
from PIL import Image
from .dataset import Dataset
from globalvars import imagenet_labels
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utilss.wordnet_utils import get_synsets

class ImageNet(Dataset):
    def __init__(self):
        super().__init__("imagenet", 1e-6, 100, imagenet_labels)
        self.train_data = []
        self.test_data = []

    def load(self, name, batch_size=64, image_size=(224, 224), num_workers=3):
        dataset_path = os.path.abspath(os.path.join("data", "datasets", name))
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist")
            return False
        
        print(f"Loading dataset from: {dataset_path}")

        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')

        # Load train and test images in batches
        start_time = time.time()
        self.train_data = list(self.load_images_in_batches(train_path, batch_size, image_size, num_workers))
        self.test_data = list(self.load_images_in_batches(test_path, batch_size, image_size, num_workers))
        end_time = time.time()

        print(f"Loaded {len(self.train_data)} training batches and {len(self.test_data)} testing batches")
        print(f"Total loading time: {end_time - start_time:.2f} seconds")
        
        return True

    def load_images_in_batches(self, folder, batch_size, image_size, num_workers):
        images = []
        labels = []
        batch_count = 0
        total_images = 0

        def load_image(image_path, class_folder):
            try:
                img = Image.open(image_path).convert('RGB')  # Load image
                img = img.resize(image_size)  # Resize image
                label = get_synsets(class_folder)  # Get class name
                return np.array(img), label
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                return None, None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for class_folder in os.listdir(folder):
                class_path = os.path.join(folder, class_folder)
                if os.path.isdir(class_path):  # Ensure it's a folder
                    for image_file in os.listdir(class_path):
                        image_path = os.path.join(class_path, image_file)
                        futures.append(executor.submit(load_image, image_path, class_folder))

            for future in as_completed(futures):
                img, label = future.result()
                if img is not None and label is not None:
                    images.append(img)
                    labels.append(label)
                    total_images += 1
                    if len(images) == batch_size:
                        batch_count += 1
                        yield np.array(images), np.array(labels)
                        print(f"Loaded batch {batch_count} from {folder}")
                        images = []
                        labels = []

        if images:
            yield np.array(images), np.array(labels)
            batch_count += 1
            print(f"Loaded batch {batch_count} from {folder}")

        print(f"Total images loaded from {folder}: {total_images}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {batch_count}")

    def get_train_image_by_id(self, image_id):
        # Implement the logic to get an image by its ID from the ImageNet dataset
        print(f"Getting image with ID: {image_id}")
        data = self.train_data
        batch_index = image_id // len(data[0][0])
        image_index = image_id % len(data[0][0])
        image = data[batch_index][0][image_index]
        label = data[batch_index][1][image_index]
        return image, label
    
    def get_test_image_by_id(self, image_id):
        # Implement the logic to get an image by its ID from the ImageNet dataset
        print(f"Getting image with ID: {image_id}")
        data = self.test_data
        batch_index = image_id // len(data[0][0])
        image_index = image_id % len(data[0][0])
        image = data[batch_index][0][image_index]
        label = data[batch_index][1][image_index]
        return image, label