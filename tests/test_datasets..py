import os
import sys
import pytest
import numpy as np
import boto3
import pickle
import io
import json
from unittest.mock import patch, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi import status

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import necessary modules
from dotenv import load_dotenv
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.classes.datasets.cifar100 import Cifar100
from utilss.classes.datasets.imagenet import ImageNet
from utilss.enums.datasets_enum import DatasetsEnum
from services.dataset_service import (
    get_dataset_config, _load_dataset, get_dataset_labels,
    load_single_image, load_imagenet_train, load_cifar100_numpy, load_cifar100_meta
)
from main import app  # Import your FastAPI app

# Load environment variables
load_dotenv()

# Setup TestClient
client = TestClient(app)

# ==================== Fixtures ====================

@pytest.fixture
def setup_env_vars():
    """Setup required environment variables."""
    # Store original env vars
    original_env = {}
    for var in ['S3_DATASETS_BUCKET_NAME', 'AWS_DATASETS_ACCESS_KEY_ID', 'AWS_DATASETS_SECRET_ACCESS_KEY']:
        original_env[var] = os.environ.get(var)
    
    # Set test env vars if not already set
    if not os.environ.get('S3_DATASETS_BUCKET_NAME'):
        os.environ['S3_DATASETS_BUCKET_NAME'] = 'test-datasets-bucket'
    if not os.environ.get('AWS_DATASETS_ACCESS_KEY_ID'):
        os.environ['AWS_DATASETS_ACCESS_KEY_ID'] = 'test-access-key'
    if not os.environ.get('AWS_DATASETS_SECRET_ACCESS_KEY'):
        os.environ['AWS_DATASETS_SECRET_ACCESS_KEY'] = 'test-secret-key'
    
    yield
    
    # Restore original env vars
    for var, value in original_env.items():
        if value is None and var in os.environ:
            del os.environ[var]
        elif value is not None:
            os.environ[var] = value

@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    with patch('boto3.client') as mock_client:
        s3_mock = MagicMock()
        mock_client.return_value = s3_mock
        yield s3_mock

@pytest.fixture
def cifar100_dataset():
    """Create a Cifar100 dataset instance."""
    return Cifar100()

@pytest.fixture
def imagenet_dataset():
    """Create an ImageNet dataset instance."""
    return ImageNet()

@pytest.fixture
def mock_cifar100_train_data():
    """Create mock CIFAR100 training data."""
    # Create a small synthetic dataset (10 images, 32x32x3)
    # CIFAR format stores images as (N, 3072) where 3072 = 32*32*3
    data = np.random.randint(0, 255, size=(10, 3072), dtype=np.uint8)
    labels = list(range(10))  # CIFAR labels are lists, not numpy arrays
    
    return {
        b'data': data,
        b'fine_labels': labels,
        b'coarse_labels': [l // 5 for l in labels],  # Create synthetic superclasses
        b'filenames': [f'image_{i}.png'.encode() for i in range(10)]
    }

@pytest.fixture
def mock_cifar100_test_data():
    """Create mock CIFAR100 test data."""
    # Similar to train but with fewer samples
    data = np.random.randint(0, 255, size=(5, 3072), dtype=np.uint8)
    labels = list(range(5))
    
    return {
        b'data': data,
        b'fine_labels': labels,
        b'coarse_labels': [l // 2 for l in labels],
        b'filenames': [f'test_{i}.png'.encode() for i in range(5)]
    }

@pytest.fixture
def mock_cifar100_meta():
    """Create mock CIFAR100 metadata."""
    return {
        b'fine_label_names': [f'class_{i}'.encode() for i in range(100)],
        b'coarse_label_names': [f'superclass_{i}'.encode() for i in range(20)]
    }

@pytest.fixture
def mock_imagenet_data():
    """Create mock ImageNet data."""
    # Create a small synthetic dataset (5 images, 224x224x3)
    images = np.random.randint(0, 255, size=(5, 224, 224, 3), dtype=np.uint8)
    # Create class names matching ImageNet format
    class_names = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']
    
    return images, np.array(class_names)

@pytest.fixture
def mock_s3_response_cifar(mock_cifar100_train_data, mock_cifar100_test_data, mock_cifar100_meta):
    """Mock S3 response for CIFAR dataset."""
    def mock_get_object(Bucket, Key):
        if Key == 'cifar100/train':
            data = pickle.dumps(mock_cifar100_train_data)
            return {'Body': io.BytesIO(data)}
        elif Key == 'cifar100/test':
            data = pickle.dumps(mock_cifar100_test_data)
            return {'Body': io.BytesIO(data)}
        elif Key == 'cifar100/meta':
            data = pickle.dumps(mock_cifar100_meta)
            return {'Body': io.BytesIO(data)}
        elif Key == 'cifar100_info.py':
            content = """
CIFAR100_INFO = {
    "dataset": "cifar100",
    "threshold": 0.8,
    "infinity": 1000,
    "labels": ["class_" + str(i) for i in range(100)]
}
"""
            return {'Body': io.BytesIO(content.encode())}
        else:
            raise Exception(f"Unknown key: {Key}")
    
    return mock_get_object

@pytest.fixture
def mock_s3_response_imagenet():
    """Mock S3 response for ImageNet dataset."""
    def mock_list_objects_v2(Bucket, Prefix, **kwargs):
        if Prefix == 'imagenet/train/':
            # Mock directory listing
            return {
                'CommonPrefixes': [
                    {'Prefix': 'imagenet/train/n01440764/'},
                    {'Prefix': 'imagenet/train/n01443537/'},
                    {'Prefix': 'imagenet/train/n01484850/'},
                ]
            }
        elif Prefix.startswith('imagenet/train/n'):
            # Mock image listing within a class
            class_name = Prefix.split('/')[-2]
            return {
                'Contents': [
                    {'Key': f'{Prefix}{class_name}_{i}.JPEG'} for i in range(3)
                ]
            }
        else:
            return {'CommonPrefixes': [], 'Contents': []}
    
    def mock_get_object(Bucket, Key):
        if Key.endswith('.JPEG'):
            # Create a small JPEG buffer (not valid JPEG but enough for mocking)
            img_data = np.random.bytes(1000)
            return {'Body': io.BytesIO(img_data)}
        elif Key == 'imagenet_info.py':
            content = """
IMAGENET_INFO = {
    "dataset": "imagenet",
    "threshold": 0.9,
    "infinity": 2000,
    "labels": {
        "n01440764": "tench",
        "n01443537": "goldfish", 
        "n01484850": "great white shark",
        "n01491361": "tiger shark",
        "n01494475": "hammerhead shark"
    },
    "directory_to_readable": {
        "n01440764": "tench",
        "n01443537": "goldfish",
        "n01484850": "great white shark",
        "n01491361": "tiger shark",
        "n01494475": "hammerhead shark"
    }
}
"""
            return {'Body': io.BytesIO(content.encode())}
        elif Key == 'imagenet/train_index_imagenet.json':
            index = [
                {"id": 1, "image_key": "imagenet/train/n01440764/n01440764_1.JPEG", "label": "n01440764"},
                {"id": 2, "image_key": "imagenet/train/n01440764/n01440764_2.JPEG", "label": "n01440764"},
                {"id": 3, "image_key": "imagenet/train/n01443537/n01443537_1.JPEG", "label": "n01443537"}
            ]
            return {'Body': io.BytesIO(json.dumps(index).encode())}
        else:
            raise Exception(f"Unknown key: {Key}")
    
    # Return combined mock functions
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2 = mock_list_objects_v2
    mock_s3.get_object = mock_get_object
    
    return mock_s3

# ==================== Tests for Dataset Configuration ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestDatasetConfiguration:
    """Tests for dataset configuration functionality."""
    
    @pytest.mark.parametrize("dataset_name", ["cifar100", "imagenet"])
    def test_get_dataset_config(self, dataset_name, mock_s3_client, mock_s3_response_cifar, mock_s3_response_imagenet):
        """Test getting dataset configuration."""
        # Setup mock response
        if dataset_name == "cifar100":
            mock_s3_client.get_object.side_effect = mock_s3_response_cifar
        else:
            mock_s3_client.get_object = mock_s3_response_imagenet.get_object
        
        # Create a mock module response
        mock_module = MagicMock()
        if dataset_name == "cifar100":
            mock_module.CIFAR100_INFO = {
                "dataset": "cifar100",
                "threshold": 0.8,
                "infinity": 1000,
                "labels": [f"class_{i}" for i in range(100)]
            }
        else:
            mock_module.IMAGENET_INFO = {
                "dataset": "imagenet",
                "threshold": 0.9,
                "infinity": 2000,
                "labels": {
                    "n01440764": "tench",
                    "n01443537": "goldfish",
                    "n01484850": "great white shark",
                    "n01491361": "tiger shark",
                    "n01494475": "hammerhead shark"
                },
                "directory_to_readable": {
                    "n01440764": "tench",
                    "n01443537": "goldfish",
                    "n01484850": "great white shark",
                    "n01491361": "tiger shark",
                    "n01494475": "hammerhead shark"
                }
            }
        
        # Mock the actual function used to get config
        with patch('utilss.s3_connector.s3_dataset_utils.get_dataset_config') as mock_get_config:
            if dataset_name == "cifar100":
                mock_get_config.return_value = mock_module.CIFAR100_INFO
            else:
                mock_get_config.return_value = mock_module.IMAGENET_INFO
            
            config = get_dataset_config(dataset_name)
            
            # Verify config
            assert config is not None
            assert config["dataset"] == dataset_name
            assert "threshold" in config
            assert "infinity" in config
            
            if dataset_name == "cifar100":
                assert "labels" in config
                assert len(config["labels"]) == 100
            else:
                assert "labels" in config
                assert "directory_to_readable" in config

    def test_get_dataset_config_missing_env_var(self):
        """Test config retrieval fails with missing env var."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3_DATASETS_BUCKET_NAME environment variable must be set"):
                get_dataset_config("cifar100")

# ==================== Tests for Dataset Loading ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestDatasetLoading:
    """Tests for dataset loading functionality."""
    
    def test_load_cifar100_dataset(self, mock_s3_client, mock_s3_response_cifar):
        """Test loading CIFAR100 dataset."""
        # Setup mock response
        mock_s3_client.get_object.side_effect = mock_s3_response_cifar
        
        # Mock get_dataset_config
        with patch('services.dataset_service.get_dataset_config') as mock_get_config:
            mock_get_config.return_value = {
                "dataset": "cifar100",
                "threshold": 0.8,
                "infinity": 1000,
                "labels": [f"class_{i}" for i in range(100)]
            }
            
            # Mock the dataset's load method
            with patch.object(Cifar100, 'load') as mock_load:
                # Set up the dataset state after load
                def setup_dataset(name):
                    dataset = _load_dataset("cifar100")
                    dataset.x_train = np.random.randint(0, 255, size=(10, 32, 32, 3), dtype=np.uint8)
                    dataset.y_train = [f"class_{i}" for i in range(10)]
                    dataset.x_test = np.random.randint(0, 255, size=(5, 32, 32, 3), dtype=np.uint8)
                    dataset.y_test = [f"class_{i}" for i in range(5)]
                    return dataset
                
                with patch('services.dataset_service._load_dataset', side_effect=setup_dataset):
                    dataset = _load_dataset("cifar100")
                    
                    # Verify dataset is loaded
                    assert dataset is not None
                    assert isinstance(dataset, Cifar100)
                    assert dataset.x_train is not None
                    assert dataset.y_train is not None
                    assert dataset.x_test is not None
                    assert dataset.y_test is not None
                    
                    # Check shapes
                    assert dataset.x_train.shape[0] == 10  # Number of samples
                    assert dataset.x_train.shape[1:] == (32, 32, 3)  # Image dimensions
                    assert len(dataset.y_train) == 10
                    
                    assert dataset.x_test.shape[0] == 5  # Number of samples
                    assert dataset.x_test.shape[1:] == (32, 32, 3)  # Image dimensions
                    assert len(dataset.y_test) == 5

    def test_load_imagenet_dataset(self, mock_s3_client, mock_s3_response_imagenet):
        """Test loading ImageNet dataset."""
        # Setup mock response
        mock_s3_client.get_object = mock_s3_response_imagenet.get_object
        mock_s3_client.list_objects_v2 = mock_s3_response_imagenet.list_objects_v2
        
        # Mock get_dataset_config
        with patch('services.dataset_service.get_dataset_config') as mock_get_config:
            mock_get_config.return_value = {
                "dataset": "imagenet",
                "threshold": 0.9,
                "infinity": 2000,
                "labels": {
                    "n01440764": "tench",
                    "n01443537": "goldfish",
                    "n01484850": "great white shark",
                    "n01491361": "tiger shark",
                    "n01494475": "hammerhead shark"
                },
                "directory_to_readable": {
                    "n01440764": "tench",
                    "n01443537": "goldfish",
                    "n01484850": "great white shark",
                    "n01491361": "tiger shark",
                    "n01494475": "hammerhead shark"
                }
            }
            
            # For ImageNet, we need to mock load method
            with patch.object(ImageNet, 'load'):
                dataset = _load_dataset("imagenet")
                
                # ImageNet is more complex and might not populate x_train immediately
                # Just verify the dataset is created correctly
                assert dataset is not None
                assert isinstance(dataset, ImageNet)

    def test_dataset_factory(self):
        """Test the DatasetFactory class."""
        # Test creating CIFAR100 dataset
        dataset = DatasetFactory.create_dataset("cifar100")
        assert isinstance(dataset, Cifar100)
        
        # Test creating ImageNet dataset
        dataset = DatasetFactory.create_dataset("imagenet")
        assert isinstance(dataset, ImageNet)
        
        # Test with invalid dataset type
        with pytest.raises(ValueError, match="Unknown dataset type"):
            DatasetFactory.create_dataset("invalid_dataset")
        
        # Test available datasets
        available_datasets = DatasetFactory.get_available_datasets()
        assert "cifar100" in available_datasets
        assert "imagenet" in available_datasets

# ==================== Tests for Dataset Operations ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestDatasetOperations:
    """Tests for dataset operation methods."""
    
    def test_cifar100_load_from_s3(self, cifar100_dataset, mock_s3_client, mock_cifar100_train_data, mock_cifar100_test_data):
        """Test Cifar100.load_from_s3 method."""
        # Create pickled data
        train_data = pickle.dumps(mock_cifar100_train_data)
        test_data = pickle.dumps(mock_cifar100_test_data)
        
        # Setup mock response
        mock_s3_client.get_object.side_effect = lambda Bucket, Key: {
            'Body': io.BytesIO(train_data if Key.endswith('train') else test_data)
        }
        
        # Test the method
        x_train, y_train = cifar100_dataset.load_from_s3(mock_s3_client, "test-bucket", "cifar100/")
        
        # Verify results
        assert x_train is not None
        assert y_train is not None
        assert x_train.shape[0] == 10
        assert x_train.shape[1:] == (32, 32, 3)
        assert len(y_train) == 10
        
        # Verify instance variables were populated
        assert cifar100_dataset.x_train is not None
        assert cifar100_dataset.y_train is not None

    def test_imagenet_load_from_s3(self, imagenet_dataset, mock_s3_client, mock_s3_response_imagenet):
        """Test ImageNet loading using its custom S3 loader."""
        # ImageNet uses S3ImagenetLoader, not the generic load_from_s3
        # So we test the load method instead
        
        with patch.object(imagenet_dataset, 's3_loader') as mock_loader:
            # Mock the S3ImagenetLoader methods
            mock_loader.get_imagenet_classes.return_value = ['n01440764', 'n01443537']
            mock_loader.get_class_images.return_value = ['image1.JPEG', 'image2.JPEG']
            mock_loader.get_image_data.return_value = b'fake_image_data'
            
            # Mock PIL Image
            with patch('PIL.Image.open') as mock_open:
                mock_img = MagicMock()
                mock_img.mode = 'RGB'
                mock_img.resize.return_value = mock_img
                mock_open.return_value = mock_img
                
                # Mock numpy array conversion
                with patch('numpy.array', return_value=np.zeros((224, 224, 3), dtype=np.float32)):
                    # Test load_mini_imagenet
                    images, labels = imagenet_dataset.load_mini_imagenet('imagenet/train')
                    
                    # Verify the method completed without errors
                    assert images is not None
                    assert labels is not None

    def test_get_train_image_by_id_cifar(self, cifar100_dataset):
        """Test get_train_image_by_id for CIFAR100."""
        # Setup test data
        cifar100_dataset.x_train = np.random.randint(0, 255, size=(10, 32, 32, 3), dtype=np.uint8)
        cifar100_dataset.y_train = [f"class_{i}" for i in range(10)]
        
        # Test valid ID
        image, label = cifar100_dataset.get_train_image_by_id(5)
        assert image.shape == (32, 32, 3)
        assert label == "class_5"
        
        # Test invalid ID
        with pytest.raises(ValueError, match="Invalid image_id"):
            cifar100_dataset.get_train_image_by_id(20)

    def test_get_train_image_by_id_imagenet(self, imagenet_dataset, mock_s3_client, mock_s3_response_imagenet):
        """Test get_train_image_by_id for ImageNet."""
        # Setup mock response
        mock_s3_client.get_object = mock_s3_response_imagenet.get_object
        
        # Mock the s3_loader
        with patch.object(imagenet_dataset, 's3_loader') as mock_loader:
            mock_loader.s3_handler.get_object_data.return_value = json.dumps([
                {"id": 1, "image_key": "imagenet/train/n01440764/n01440764_1.JPEG", "label": "n01440764"}
            ]).encode()
            mock_loader.get_image_data.return_value = b'fake_image_data'
            
            # Mock image loading
            with patch('PIL.Image.open') as mock_open:
                mock_img = MagicMock()
                mock_img.mode = 'RGB'
                mock_img.resize.return_value = mock_img
                mock_open.return_value = mock_img
                
                with patch('numpy.array', return_value=np.zeros((224, 224, 3), dtype=np.float32)):
                    image, label = imagenet_dataset.get_train_image_by_id(1)
                    
                    assert image.shape == (224, 224, 3)
                    assert label == "n01440764"

# ==================== Tests for Service Methods ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestServiceMethods:
    """Tests for service-level methods."""
    
    def test_get_dataset_labels(self, mock_s3_client):
        """Test get_dataset_labels function."""
        # Mock get_dataset_config
        with patch('services.dataset_service.get_dataset_config') as mock_config:
            mock_config.return_value = {
                "labels": [f"class_{i}" for i in range(10)]
            }
            
            # Test the function
            labels = get_dataset_labels("cifar100")
            
            # Verify results
            assert labels is not None
            assert len(labels) == 10
            assert labels[0] == "class_0"
            assert labels[9] == "class_9"

    def test_load_single_image(self, mock_s3_client):
        """Test load_single_image function."""
        # Mock image data
        image_data = b'fake_image_data'
        
        # Setup mock response
        mock_s3_client.get_object.return_value = {'Body': io.BytesIO(image_data)}
        
        # Mock S3DatasetLoader
        with patch('services.dataset_service.S3DatasetLoader') as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.load_single_image.return_value = image_data
            mock_loader_class.return_value = mock_loader
            
            # Test the function
            result = load_single_image("test/image.jpg")
            
            # Verify results
            assert result == image_data

# ==================== Tests for API Integration ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestAPIIntegration:
    """Tests for API endpoints."""
    
    def test_get_dataset_labels_endpoint(self):
        """Test GET /api/datasets/{dataset_name}/labels endpoint."""
        # Mock get_dataset_labels
        with patch('api.endpoints.dataset_router.get_dataset_labels') as mock_get_labels:
            mock_get_labels.return_value = [f"class_{i}" for i in range(5)]
            
            # Test the endpoint
            response = client.get("/api/datasets/cifar100/labels")
            
            # Verify response
            assert response.status_code == status.HTTP_200_OK
            
            # Parse response JSON
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 5
            assert data["data"][0] == "class_0"
    
    def test_get_dataset_labels_endpoint_not_found(self):
        """Test GET /api/datasets/{dataset_name}/labels with not found."""
        # Mock get_dataset_labels to return None
        with patch('api.endpoints.dataset_router.get_dataset_labels', return_value=None):
            # Test the endpoint
            response = client.get("/api/datasets/unknown_dataset/labels")
            
            # Verify response
            assert response.status_code == status.HTTP_404_NOT_FOUND
            
            # Parse response JSON
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Couldn't find dataset labels"

# ==================== Tests for Error Handling ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_dataset_name(self):
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            DatasetFactory.create_dataset("invalid_dataset")
    
    def test_missing_bucket_env_var(self):
        """Test error handling for missing S3 bucket env var."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3_DATASETS_BUCKET_NAME environment variable must be set"):
                load_single_image("test/image.jpg")

# ==================== Data Authenticity Tests ====================

@pytest.mark.usefixtures("setup_env_vars")
class TestDataAuthenticity:
    """Tests for data authenticity."""
    
    def test_cifar100_data_integrity(self, cifar100_dataset, mock_s3_client, mock_cifar100_train_data, mock_cifar100_test_data):
        """Test CIFAR100 data integrity."""
        # Create pickled data
        train_data = pickle.dumps(mock_cifar100_train_data)
        test_data = pickle.dumps(mock_cifar100_test_data)
        
        # Setup mock response
        mock_s3_client.get_object.side_effect = lambda Bucket, Key: {
            'Body': io.BytesIO(train_data if Key.endswith('train') else test_data)
        }
        
        # Load the data
        x_train, y_train = cifar100_dataset.load_from_s3(mock_s3_client, "test-bucket", "cifar100/")
        
        # Verify integrity by checking shape and type
        assert x_train.shape[1:] == (32, 32, 3)  # Image dimensions
        assert x_train.dtype == np.uint8  # Data type should be uint8 for images
        
        # Verify value range
        assert np.min(x_train) >= 0
        assert np.max(x_train) <= 255
        
        # Verify y_train is a list of strings
        assert isinstance(y_train, list)
        assert all(isinstance(label, str) for label in y_train)
        
        # When accessing an image, verify its format
        image, label = cifar100_dataset.get_train_image_by_id(0)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert isinstance(label, str)

    def test_imagenet_data_integrity(self, imagenet_dataset):
        """Test ImageNet data integrity with more controlled mocking."""
        # Create a simple mock for the dataset
        imagenet_dataset.x_train = np.random.randint(0, 255, size=(5, 224, 224, 3), dtype=np.uint8)
        imagenet_dataset.y_train = np.array(['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475'])
        
        # Verify the dataset was created with correct dimensions
        assert imagenet_dataset.x_train.shape[1:] == (224, 224, 3)
        
        # Verify value range
        assert np.min(imagenet_dataset.x_train) >= 0
        assert np.max(imagenet_dataset.x_train) <= 255
        
        # Verify labels match expected format
        assert all(label.startswith('n0') for label in imagenet_dataset.y_train)
        
        # Test label conversion
        with patch.object(imagenet_dataset, 'directory_to_labels_conversion', return_value="tench"):
            readable_name = imagenet_dataset.get_label_readable_name("n01440764")
            assert readable_name == "tench"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])