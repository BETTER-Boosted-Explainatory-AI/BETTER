import dotenv
dotenv.load_dotenv()

import unittest
from unittest.mock import MagicMock, patch
import os
import numpy as np
import io
import json
from PIL import Image
from fastapi.testclient import TestClient

from utilss.classes.datasets.cifar100 import Cifar100
from utilss.classes.datasets.imagenet import ImageNet
from utilss.classes.datasets.dataset_factory import DatasetFactory
from services.dataset_service import get_dataset_labels, _load_dataset, load_single_image
from app import app 

client = TestClient(app)


class TestDatasetIntegration(unittest.TestCase):
    @patch('boto3.client')
    def setUp(self, mock_boto3_client):
        # Set up mock S3 client
        self.mock_s3 = MagicMock()
        mock_boto3_client.return_value = self.mock_s3
        
        # Set environment variable for S3 bucket
        os.environ['S3_DATASETS_BUCKET_NAME'] = 'test-bucket'
        
        # Create common test data
        self.train_data = {
            b'data': np.random.randint(0, 255, (10, 3072), dtype=np.uint8),
            b'fine_labels': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        }
        self.test_data = {
            b'data': np.random.randint(0, 255, (5, 3072), dtype=np.uint8),
            b'fine_labels': [0, 1, 2, 3, 4]
        }
        
        # Create mock config data
        self.cifar100_config = {
            "dataset": "cifar100",
            "threshold": 0.5,
            "infinity": 100,
            "labels": ["apple", "bear", "cat", "dog", "elephant"]
        }
        
        self.imagenet_config = {
            "dataset": "imagenet",
            "threshold": 0.5,
            "infinity": 100,
            "labels": ["n01440764", "n01443537", "n01484850"],
            "directory_to_readable": {
                "n01440764": "tench",
                "n01443537": "goldfish",
                "n01484850": "great white shark"
            }
        }
        
        # Set up patches
        self.patches = []
    
    def tearDown(self):
        # Remove environment variable
        if 'S3_DATASETS_BUCKET_NAME' in os.environ:
            del os.environ['S3_DATASETS_BUCKET_NAME']
        
        # Clean up patches
        for patcher in self.patches:
            patcher.stop()
    
    def add_patch(self, *args, **kwargs):
        """Helper method to add a patch and start it."""
        patcher = patch(*args, **kwargs)
        self.patches.append(patcher)
        return patcher.start()
    
    @patch('services.dataset_service.get_dataset_config')
    @patch('services.dataset_service.DatasetFactory')
    def test_load_dataset_cifar100(self, mock_factory, mock_get_config):
        """Test loading CIFAR-100 dataset through the service layer."""
        # Configure mocks
        mock_get_config.return_value = self.cifar100_config
        
        # Create a mock CIFAR-100 dataset
        mock_cifar100 = MagicMock(spec=Cifar100)
        mock_factory.create_dataset.return_value = mock_cifar100
        
        # Call the function
        _load_dataset("cifar100")
        
        # Verify correct interactions
        mock_get_config.assert_called_once_with("cifar100")
        mock_factory.create_dataset.assert_called_once_with("cifar100")
        mock_cifar100.load.assert_called_once_with("cifar100")
    
    @patch('utilss.s3_connector.s3_dataset_utils.get_dataset_config')
    def test_get_dataset_labels_integration(self, mock_get_config):
        """Test getting dataset labels through the service layer."""
        # Configure mocks
        mock_get_config.return_value = self.cifar100_config
        
        # Call the function
        labels = get_dataset_labels("cifar100")
        
        # Verify the results
        self.assertEqual(labels, ["apple", "bear", "cat", "dog", "elephant"])
        mock_get_config.assert_called_once_with("cifar100")
    
    @patch('boto3.client')
    def test_dataset_api_endpoint_integration(self, mock_boto3):
        """Test the dataset API endpoint with mock data."""
        # Mock get_dataset_labels to return test data
        with patch('services.dataset_service.get_dataset_labels') as mock_labels:
            mock_labels.return_value = ["apple", "bear", "cat", "dog", "elephant"]
            
            # Test the endpoint
            response = client.get("/api/datasets/cifar100/labels")
            
            # Verify the response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"data": ["apple", "bear", "cat", "dog", "elephant"]})
            mock_labels.assert_called_once_with("cifar100")
    
    def test_factory_integration(self):
        """Test dataset factory integration with concrete dataset classes."""
        # Mock the dataset configurations
        with patch('utilss.s3_connector.s3_dataset_utils.get_dataset_config') as mock_get_config:
            # Configure mock to return appropriate config based on dataset name
            def config_side_effect(dataset_name):
                if dataset_name == "cifar100":
                    return self.cifar100_config
                elif dataset_name == "imagenet":
                    return self.imagenet_config
                else:
                    return None
            
            mock_get_config.side_effect = config_side_effect
            
            # Test CIFAR-100 creation
            with patch.object(Cifar100, 'load') as mock_cifar_load:
                mock_cifar_load.return_value = True
                
                # Create the dataset
                dataset = DatasetFactory.create_dataset("cifar100")
                
                # Verify it's the right type
                self.assertIsInstance(dataset, Cifar100)
                
                # Configure the dataset and verify initialization
                dataset.load("cifar100")
                mock_cifar_load.assert_called_once_with("cifar100")
            
            # Test ImageNet creation
            with patch.object(ImageNet, 'load') as mock_imagenet_load:
                mock_imagenet_load.return_value = True
                
                # Create the dataset
                dataset = DatasetFactory.create_dataset("imagenet")
                
                # Verify it's the right type
                self.assertIsInstance(dataset, ImageNet)
                
                # Configure the dataset and verify initialization
                dataset.load("imagenet")
                mock_imagenet_load.assert_called_once_with("imagenet")
    
    def test_cifar100_full_pipeline(self):
        """Test the full CIFAR-100 dataset loading pipeline."""
        # Mock get_dataset_config to return our config
        mock_get_config = self.add_patch('utilss.s3_connector.s3_dataset_utils.get_dataset_config')
        mock_get_config.return_value = self.cifar100_config
        
        # Mock unpickle_from_s3 to return test data
        mock_unpickle = self.add_patch('utilss.s3_connector.s3_dataset_utils.unpickle_from_s3')
        mock_unpickle.side_effect = lambda bucket, key: self.train_data if key == "cifar100/train" else self.test_data
        
        # Create the dataset
        cifar100 = Cifar100()
        
        # Load the dataset
        result = cifar100.load("cifar100")
        
        # Verify the results
        self.assertTrue(result)
        self.assertEqual(cifar100.x_train.shape, (10, 32, 32, 3))
        self.assertEqual(len(cifar100.y_train), 10)
        self.assertEqual(cifar100.x_test.shape, (5, 32, 32, 3))
        self.assertEqual(len(cifar100.y_test), 5)
        
        # Test one_hot_to_class_name_auto
        test_one_hot = np.zeros(5)
        test_one_hot[2] = 1
        class_name = cifar100.one_hot_to_class_name_auto(test_one_hot)
        self.assertEqual(class_name, "cat")  # Index 2 corresponds to "cat"
        
        # Test get_train_image_by_id
        image, label = cifar100.get_train_image_by_id(2)
        self.assertEqual(image.shape, (32, 32, 3))
        
        # Test get_test_image_by_id
        image, label = cifar100.get_test_image_by_id(2)
        self.assertEqual(image.shape, (32, 32, 3))
    
    def test_imagenet_pipeline_with_index(self):
        """Test the ImageNet dataset pipeline with index creation and retrieval."""
        # Mock get_dataset_config to return our config
        mock_get_config = self.add_patch('utilss.s3_connector.s3_dataset_utils.get_dataset_config')
        mock_get_config.return_value = self.imagenet_config
        
        # Mock the S3ImagenetLoader
        mock_s3_loader_class = self.add_patch('utilss.classes.datasets.imagenet.S3ImagenetLoader')
        mock_s3_loader = MagicMock()
        mock_s3_loader_class.return_value = mock_s3_loader
        
        # Mock s3_handler
        mock_s3_handler = MagicMock()
        mock_s3_loader.s3_handler = mock_s3_handler
        
        # Set up get_imagenet_classes to return class names
        mock_s3_loader.get_imagenet_classes.return_value = ["n01440764", "n01443537"]
        
        # Set up get_class_images to return images for each class
        class_images = {
            "n01440764": ["imagenet/train/n01440764/img1.jpg", "imagenet/train/n01440764/img2.jpg"],
            "n01443537": ["imagenet/train/n01443537/img1.jpg", "imagenet/train/n01443537/img2.jpg"]
        }
        mock_s3_loader.get_class_images.side_effect = lambda class_name: class_images.get(class_name, [])
        
        # Set up get_image_data to return test image data
        def get_image_data(image_key):
            img = Image.new('RGB', (32, 32), color='red')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
        mock_s3_loader.get_image_data.side_effect = get_image_data
        
        # Create a mock index
        index_data = [
            {"id": 1, "image_key": "imagenet/train/n01440764/img1.jpg", "label": "n01440764"},
            {"id": 2, "image_key": "imagenet/train/n01443537/img1.jpg", "label": "n01443537"}
        ]
        index_json = json.dumps(index_data)
        mock_s3_handler.get_object_data.return_value = index_json.encode('utf-8')
        
        # Create the dataset
        imagenet = ImageNet()
        
        # Build the index
        imagenet.build_index("train")
        
        # Verify the index was built correctly
        self.assertEqual(len(imagenet.train_index), 4)  # 2 classes * 2 images each
        
        # Test get_train_image_by_id
        image, label = imagenet.get_train_image_by_id(1)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(label, "n01440764")
        
        # Test directory_to_labels_conversion
        readable_label = imagenet.directory_to_labels_conversion("n01440764")
        self.assertEqual(readable_label, "tench")
        
        # Test get_label_readable_name
        readable_label = imagenet.get_label_readable_name("n01440764")
        self.assertEqual(readable_label, "tench")
    
    def test_fastapi_error_handling(self):
        """Test error handling in the FastAPI endpoint."""
        # Mock get_dataset_labels to return None
        with patch('services.dataset_service.get_dataset_labels') as mock_labels:
            mock_labels.return_value = None
            
            # Test the endpoint with an invalid dataset
            response = client.get("/api/datasets/invalid_dataset/labels")
            
            # Verify the response
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"detail": "Couldn't find dataset labels"})
            mock_labels.assert_called_once_with("invalid_dataset")
    
        
        # Create the dataset
        cifar100 = Cifar100()
        
        # Try to load the dataset without environment variable
        with self.assertRaises(RuntimeError):
            cifar100.load("cifar100")
        
        # Try to load a single image without environment variable
        with self.assertRaises(ValueError):
            load_single_image("test/image.jpg")


if __name__ == "__main__":
    unittest.main()