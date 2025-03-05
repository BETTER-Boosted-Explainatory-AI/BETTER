import os
from PIL import Image
from .dataset import Dataset
from globalvars import imagenet_labels

class ImageNet(Dataset):
    def __init__(self):
        super().__init__("imagenet", 1e-6, 100, imagenet_labels)
        self.train_data = []
        self.test_data = []

    def load(self, name):
        dataset_path = os.path.abspath(os.path.join("data", "datasets", name))
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist")
            return
        
        print(f"Loading dataset from: {dataset_path}")

        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')

        # Helper function to load images from a directory
        def load_images_from_folder(folder):
            images = []
            for class_folder in os.listdir(folder):
                class_path = os.path.join(folder, class_folder)
                if os.path.isdir(class_path):  # Ensure it's a folder
                    for image_file in os.listdir(class_path):
                        image_path = os.path.join(class_path, image_file)
                        try:
                            img = Image.open(image_path).convert('RGB')  # Load image
                            images.append((img, class_folder))  # Store with class label
                        except Exception as e:
                            print(f"Error loading {image_path}: {e}")
            return images

        # Load train and test images
        self.train_data = load_images_from_folder(train_path)
        self.test_data = load_images_from_folder(test_path)

        print(f"Loaded {len(self.train_data)} training images and {len(self.test_data)} testing images")

    def get_image_by_id(self, image_id):
        # Implement the logic to get an image by its ID from the CIFAR-100 dataset
        print(f"Getting image with ID: {image_id}")