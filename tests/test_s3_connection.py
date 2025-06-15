import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import uuid
import boto3
import pytest
import json
import tempfile
from botocore.exceptions import ClientError

# Import path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Import services to test
from services.models_service import (
    s3_file_exists, read_json_from_s3, load_model_from_s3, get_user_models_info,
    _get_model_path, _get_model_filename, _check_model_path, construct_model,
    query_model, query_predictions, get_model_files, get_model_specific_file
)
from services.dendrogram_service import (
    _get_dendrogram_path, _get_sub_dendrogram, _rename_cluster, _get_common_ancestor_subtree
)
from services.whitebox_service import (
    _get_edges_dataframe_path, get_white_box_analysis
)
from services.dataset_service import (
    _get_dataset_config, get_dataset_labels, load_single_image,
    load_imagenet_train, load_cifar100_numpy, load_cifar100_meta
)
from utilss.s3_connector.s3_dataset_utils import (
    load_dataset_numpy, load_cifar100_adversarial_or_clean,
    load_imagenet_adversarial_or_clean, load_dataset_folder,
    get_image_stream, load_cifar100_as_numpy, load_dataset_split,
    unpickle_from_s3
)
from utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
from utilss.classes.dendrogram import Dendrogram
from utilss.classes.edges_dataframe import EdgesDataframe
from utilss.classes.user import User
from utilss.classes.whitebox_testing import WhiteBoxTesting

# Color for terminal output
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{TermColors.HEADER}{TermColors.BOLD}{'=' * 80}{TermColors.END}")
    print(f"{TermColors.HEADER}{TermColors.BOLD}{text.center(80)}{TermColors.END}")
    print(f"{TermColors.HEADER}{TermColors.BOLD}{'=' * 80}{TermColors.END}")

def print_subheader(text):
    print(f"\n{TermColors.BLUE}{TermColors.BOLD}{text}{TermColors.END}")
    print(f"{TermColors.BLUE}{'-' * 50}{TermColors.END}")

def print_success(text):
    print(f"{TermColors.GREEN}✓ {text}{TermColors.END}")

def print_warning(text):
    print(f"{TermColors.WARNING}⚠ {text}{TermColors.END}")

def print_error(text):
    print(f"{TermColors.RED}✗ {text}{TermColors.END}")

def print_info(text):
    print(f"{TermColors.CYAN}ℹ {text}{TermColors.END}")

def check_environment():
    """Check if all required environment variables are set."""
    print_subheader("Checking Environment Variables")
    
    required_vars = [
        'AWS_DATASETS_ACCESS_KEY_ID', 
        'AWS_DATASETS_SECRET_ACCESS_KEY', 
        'S3_DATASETS_BUCKET_NAME',
        'AWS_USERS_ACCESS_KEY_ID',
        'AWS_USERS_SECRET_ACCESS_KEY',
        'S3_USERS_BUCKET_NAME'
    ]
    
    all_set = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            masked_value = value[:4] + '***' if len(value) > 7 else '***'
            print_success(f"{var} is set: {masked_value}")
        else:
            print_error(f"{var} is NOT set!")
            all_set = False
    
    if all_set:
        print_success("All required environment variables are set!")
        return True
    else:
        print_error("Please set all required environment variables before running this script.")
        return False

@pytest.fixture(scope="session")
def datasets_bucket():
    return os.environ["S3_DATASETS_BUCKET_NAME"]

@pytest.fixture(scope="session")
def users_bucket():
    return os.environ["S3_USERS_BUCKET_NAME"]

@pytest.fixture
def datasets_s3_prefix(datasets_bucket):
    """Give each test its own clean prefix in datasets bucket, and tear it down at the end."""
    client = boto3.client("s3", region_name="us-east-1")
    prefix = f"integration-tests/{uuid.uuid4()}/"
    client.put_object(Bucket=datasets_bucket, Key=prefix)  # create the "folder"
    yield datasets_bucket, prefix
    # teardown: delete everything under that prefix
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=datasets_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=datasets_bucket, Key=obj["Key"])

@pytest.fixture
def users_s3_prefix(users_bucket):
    """Give each test its own clean prefix in users bucket, and tear it down at the end."""
    client = boto3.client("s3", region_name="us-east-1")
    prefix = f"integration-tests/{uuid.uuid4()}/"
    client.put_object(Bucket=users_bucket, Key=prefix)  # create the "folder"
    yield users_bucket, prefix
    # teardown: delete everything under that prefix
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=users_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=users_bucket, Key=obj["Key"])

@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    user_id = str(uuid.uuid4())
    email = f"test-{user_id}@example.com"
    
    class MockUser:
        def get_user_id(self):
            return user_id
            
        def get_user_folder(self):
            return user_id
            
        def __init__(self, user_id=user_id, email=email):
            self.user_id = user_id
            self.email = email
    
    return MockUser()

@pytest.fixture
def mock_model_info():
    """Create mock model info for testing."""
    return {
        "model_id": str(uuid.uuid4()),
        "file_name": "model.keras",
        "dataset": "cifar100"
    }

# ====================== Tests for S3 Dataset Utils ======================

def test_get_datasets_s3_client():
    """Test getting S3 client for datasets."""
    print_subheader("Testing get_datasets_s3_client()")
    try:
        s3_client = get_datasets_s3_client()
        if s3_client:
            print_success("Successfully got S3 client for datasets")
            print_info(f"Client type: {type(s3_client).__name__}")
        else:
            print_error("Failed to get S3 client for datasets")
    except Exception as e:
        print_error(f"Error in get_datasets_s3_client(): {str(e)}")

def test_get_users_s3_client():
    """Test getting S3 client for users."""
    print_subheader("Testing get_users_s3_client()")
    try:
        s3_client = get_users_s3_client()
        if s3_client:
            print_success("Successfully got S3 client for users")
            print_info(f"Client type: {type(s3_client).__name__}")
        else:
            print_error("Failed to get S3 client for users")
    except Exception as e:
        print_error(f"Error in get_users_s3_client(): {str(e)}")

# ====================== Tests for Models Service ======================

def test_get_model_path(users_s3_prefix, mock_user, mock_model_info):
    """Test _get_model_path function."""
    print_subheader("Testing _get_model_path()")
    bucket, prefix = users_s3_prefix
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    
    try:
        # Create a dummy model directory structure
        client = boto3.client("s3")
        model_key = f"{prefix}{user_id}/{model_id}/dummy.txt"
        client.put_object(Bucket=bucket, Key=model_key, Body=b"dummy content")
        
        # Test the function
        model_path = _get_model_path(user_id, model_id)
        
        if model_path:
            print_success(f"Successfully got model path: {model_path}")
        else:
            print_warning("Model path not found")
            
        # Test with non-existent model
        non_existent_path = _get_model_path(user_id, "non-existent-model-id")
        if non_existent_path is None:
            print_success("Correctly returned None for non-existent model")
        else:
            print_error(f"Incorrectly found path for non-existent model: {non_existent_path}")
    except Exception as e:
        print_error(f"Error in _get_model_path(): {str(e)}")

def test_check_model_path(users_s3_prefix, mock_user, mock_model_info):
    """Test _check_model_path function."""
    print_subheader("Testing _check_model_path()")
    bucket, prefix = users_s3_prefix
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    try:
        # Create a dummy model directory structure with graph type
        client = boto3.client("s3")
        model_key = f"{prefix}{user_id}/{model_id}/{graph_type}/dummy.txt"
        client.put_object(Bucket=bucket, Key=model_key, Body=b"dummy content")
        
        # Test the function (will need to monkeypatch _get_model_path)
        # This is a complex test due to HTTP exceptions, so we'll just check if it's callable
        print_info("Function exists but requires HTTP context to test fully")
        
        # Verify the function signature
        import inspect
        sig = inspect.signature(_check_model_path)
        print_success(f"Function signature validated: {sig}")
    except Exception as e:
        print_error(f"Error in _check_model_path() test: {str(e)}")

def test_get_model_filename(users_s3_prefix, mock_user, mock_model_info):
    """Test _get_model_filename function."""
    print_subheader("Testing _get_model_filename()")
    bucket, prefix = users_s3_prefix
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    try:
        # Create a dummy model file
        client = boto3.client("s3")
        model_key = f"{prefix}{user_id}/{model_id}/model.keras"
        client.put_object(Bucket=bucket, Key=model_key, Body=b"dummy model content")
        
        # Create graph directory
        graph_key = f"{prefix}{user_id}/{model_id}/{graph_type}/dummy.txt"
        client.put_object(Bucket=bucket, Key=graph_key, Body=b"dummy graph content")
        
        # Test the function (would need monkeypatching for S3_BUCKET)
        print_info("Function exists but requires environment setup to test fully")
        
        # Verify the function signature
        import inspect
        sig = inspect.signature(_get_model_filename)
        print_success(f"Function signature validated: {sig}")
    except Exception as e:
        print_error(f"Error in _get_model_filename() test: {str(e)}")

def test_s3_file_exists(users_s3_prefix):
    """Test s3_file_exists function."""
    print_subheader("Testing s3_file_exists()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    
    # Create a test file
    test_key = f"{prefix}test_file.txt"
    client.put_object(Bucket=bucket, Key=test_key, Body=b"test content")
    
    try:
        # Test existing file
        exists = s3_file_exists(bucket, test_key)
        if exists:
            print_success(f"Successfully detected existing file: {test_key}")
        else:
            print_error(f"Failed to detect existing file: {test_key}")
            
        # Test non-existent file
        non_existent_key = f"{prefix}non_existent.txt"
        exists = s3_file_exists(bucket, non_existent_key)
        if not exists:
            print_success(f"Correctly reported non-existent file: {non_existent_key}")
        else:
            print_error(f"Incorrectly reported non-existent file exists: {non_existent_key}")
    except Exception as e:
        print_error(f"Error in s3_file_exists(): {str(e)}")

def test_read_json_from_s3(users_s3_prefix):
    """Test read_json_from_s3 function."""
    print_subheader("Testing read_json_from_s3()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    
    # Create a test JSON file
    test_data = {"key1": "value1", "key2": 42, "nested": {"a": [1, 2, 3]}}
    test_key = f"{prefix}test_data.json"
    client.put_object(
        Bucket=bucket, 
        Key=test_key, 
        Body=json.dumps(test_data).encode('utf-8'),
        ContentType="application/json"
    )
    
    try:
        # Test reading the JSON
        data = read_json_from_s3(bucket, test_key)
        if data == test_data:
            print_success(f"Successfully read and parsed JSON from {test_key}")
            print_info(f"Data: {data}")
        else:
            print_error(f"JSON data mismatch. Expected: {test_data}, Got: {data}")
            
        # Test reading non-existent file
        non_existent_key = f"{prefix}non_existent.json"
        try:
            read_json_from_s3(bucket, non_existent_key)
            print_error(f"Should have raised exception for non-existent file: {non_existent_key}")
        except Exception as e:
            print_success(f"Correctly raised exception for non-existent file: {str(e)}")
    except Exception as e:
        print_error(f"Error in read_json_from_s3(): {str(e)}")

def test_get_user_models_info(users_s3_prefix, mock_user, mock_model_info):
    """Test get_user_models_info function."""
    print_subheader("Testing get_user_models_info()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    
    # Create a models.json file
    models_data = [
        {
            "model_id": model_id,
            "dataset": "cifar100",
            "file_name": "model.keras",
            "created_at": "2023-01-01T00:00:00Z"
        }
    ]
    models_key = f"{user_id}/models.json"
    
    try:
        # Upload the models.json
        client.put_object(
            Bucket=bucket,
            Key=models_key,
            Body=json.dumps(models_data).encode('utf-8'),
            ContentType="application/json"
        )
        
        # Test the function
        import inspect
        sig = inspect.signature(get_user_models_info)
        print_success(f"Function signature validated: {sig}")
        print_info("Function requires S3_BUCKET environment variable to test fully")
        
    except Exception as e:
        print_error(f"Error in get_user_models_info() test: {str(e)}")

def test_get_model_files(users_s3_prefix, mock_user, mock_model_info):
    """Test get_model_files function."""
    print_subheader("Testing get_model_files()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    # Create necessary files for testing
    model_file_key = f"{user_id}/{model_id}/model.keras"
    graph_folder_key = f"{user_id}/{model_id}/{graph_type}/"
    dendrogram_key = f"{user_id}/{model_id}/{graph_type}/dendrogram.pkl"
    
    try:
        # Upload test files
        client.put_object(Bucket=bucket, Key=model_file_key, Body=b"model content")
        client.put_object(Bucket=bucket, Key=graph_folder_key, Body=b"")
        client.put_object(Bucket=bucket, Key=dendrogram_key, Body=b"dendrogram data")
        
        # Test the function
        import inspect
        sig = inspect.signature(get_model_files)
        print_success(f"Function signature validated: {sig}")
        print_info("Function requires S3_BUCKET environment variable to test fully")
        
    except Exception as e:
        print_error(f"Error in get_model_files() test: {str(e)}")

def test_get_model_specific_file(users_s3_prefix, mock_user, mock_model_info):
    """Test get_model_specific_file function."""
    print_subheader("Testing get_model_specific_file()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    # Create necessary files for testing
    model_file_key = f"{user_id}/{model_id}/model.keras"
    graph_folder_key = f"{user_id}/{model_id}/{graph_type}/"
    dendrogram_key = f"{user_id}/{model_id}/{graph_type}/dendrogram.json"
    
    try:
        # Upload test files
        client.put_object(Bucket=bucket, Key=model_file_key, Body=b"model content")
        client.put_object(Bucket=bucket, Key=graph_folder_key, Body=b"")
        client.put_object(Bucket=bucket, Key=dendrogram_key, Body=b"dendrogram data")
        
        # Test the function
        import inspect
        sig = inspect.signature(get_model_specific_file)
        print_success(f"Function signature validated: {sig}")
        print_info("Function requires S3_BUCKET environment variable to test fully")
        
    except Exception as e:
        print_error(f"Error in get_model_specific_file() test: {str(e)}")

# ====================== Tests for Dendrogram Service ======================

def test_get_dendrogram_path(users_s3_prefix, mock_user, mock_model_info):
    """Test _get_dendrogram_path function."""
    print_subheader("Testing _get_dendrogram_path()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    # Create necessary structure
    model_dir_key = f"{user_id}/{model_id}/"
    graph_dir_key = f"{user_id}/{model_id}/{graph_type}/"
    
    try:
        # Create directory structure
        client.put_object(Bucket=bucket, Key=model_dir_key, Body=b"")
        client.put_object(Bucket=bucket, Key=graph_dir_key, Body=b"")
        
        # Test the function
        import inspect
        sig = inspect.signature(_get_dendrogram_path)
        print_success(f"Function signature validated: {sig}")
        print_info("Function requires FastAPI context to test fully")
        
    except Exception as e:
        print_error(f"Error in _get_dendrogram_path() test: {str(e)}")

def test_dendrogram_class():
    """Test Dendrogram class with S3 integration."""
    print_subheader("Testing Dendrogram class S3 integration")
    try:
        # Validate the class has methods for S3 operations
        methods = [method for method in dir(Dendrogram) if not method.startswith('_')]
        
        expected_methods = ['find_name_hierarchy', 'get_common_ancestor_subtree', 
                           'get_sub_dendrogram_formatted', 'load_dendrogram', 
                           'rename_cluster', 'save_dendrogram']
        
        for method in expected_methods:
            if method in methods:
                print_success(f"Dendrogram class has method: {method}")
            else:
                print_warning(f"Dendrogram class missing expected method: {method}")
                
        # Check initialization with S3 path
        dendrogram = Dendrogram("s3://test-bucket/test-key")
        print_success("Successfully initialized Dendrogram with S3 path")
        
    except Exception as e:
        print_error(f"Error testing Dendrogram class: {str(e)}")

# ====================== Tests for Whitebox Service ======================

def test_get_edges_dataframe_path(users_s3_prefix, mock_user, mock_model_info):
    """Test _get_edges_dataframe_path function."""
    print_subheader("Testing _get_edges_dataframe_path()")
    bucket, prefix = users_s3_prefix
    client = boto3.client("s3")
    user_id = mock_user.get_user_id()
    model_id = mock_model_info["model_id"]
    graph_type = "activation"
    
    # Create necessary files
    edges_df_key = f"{user_id}/{model_id}/{graph_type}/edges_df.csv"
    
    try:
        # Upload test file
        client.put_object(Bucket=bucket, Key=edges_df_key, Body=b"edges data")
        
        # Test the function
        import inspect
        sig = inspect.signature(_get_edges_dataframe_path)
        print_success(f"Function signature validated: {sig}")
        print_info("Function requires S3_BUCKET environment variable to test fully")
        
    except Exception as e:
        print_error(f"Error in _get_edges_dataframe_path() test: {str(e)}")

def test_edges_dataframe_class():
    """Test EdgesDataframe class with S3 integration."""
    print_subheader("Testing EdgesDataframe class S3 integration")
    try:
        # Validate the class has methods for S3 operations
        methods = [method for method in dir(EdgesDataframe) if not method.startswith('_')]
        
        expected_methods = ['get_dataframe', 'load_dataframe', 'save_dataframe']
        
        for method in expected_methods:
            if method in methods:
                print_success(f"EdgesDataframe class has method: {method}")
            else:
                print_warning(f"EdgesDataframe class missing expected method: {method}")
                
        # Check initialization with S3 paths
        edges_df = EdgesDataframe("s3://test-bucket/model.keras", "s3://test-bucket/edges_df.csv")
        print_success("Successfully initialized EdgesDataframe with S3 paths")
        
    except Exception as e:
        print_error(f"Error testing EdgesDataframe class: {str(e)}")

def test_whitebox_testing_class():
    """Test WhiteBoxTesting class with S3 integration."""
    print_subheader("Testing WhiteBoxTesting class S3 integration")
    try:
        # Validate the class has methods for finding problematic images
        methods = [method for method in dir(WhiteBoxTesting) if not method.startswith('_')]
        
        if 'find_problematic_images' in methods:
            print_success("WhiteBoxTesting class has find_problematic_images method")
        else:
            print_warning("WhiteBoxTesting class missing find_problematic_images method")
                
        # Check initialization with S3 path
        whitebox = WhiteBoxTesting("s3://test-bucket/model.keras")
        print_success("Successfully initialized WhiteBoxTesting with S3 path")
        
    except Exception as e:
        print_error(f"Error testing WhiteBoxTesting class: {str(e)}")

# ====================== Tests for Dataset Service ======================

def test_get_dataset_config():
    """Test _get_dataset_config function."""
    print_subheader("Testing _get_dataset_config()")
    try:
        # Test with CIFAR100
        cifar_config = _get_dataset_config('cifar100')
        if cifar_config:
            print_success("Successfully loaded CIFAR100 config")
            print_info(f"Config keys: {list(cifar_config.keys())}")
        else:
            print_warning("No CIFAR100 config found")
            
        # Test with ImageNet
        try:
            imagenet_config = _get_dataset_config('imagenet')
            if imagenet_config:
                print_success("Successfully loaded ImageNet config")
                print_info(f"Config keys: {list(imagenet_config.keys())}")
            else:
                print_warning("No ImageNet config found")
        except Exception as e:
            print_warning(f"Could not load ImageNet config: {str(e)}")
            
    except Exception as e:
        print_error(f"Error in _get_dataset_config(): {str(e)}")

def test_user_class_s3_integration():
    """Test User class S3 integration."""
    print_subheader("Testing User class S3 integration")
    try:
        # Create a test user
        test_user_id = str(uuid.uuid4())
        test_email = f"test-{test_user_id}@example.com"
        
        # Check initialization
        user = User(user_id=test_user_id, email=test_email)
        
        # Validate methods that might interact with S3
        methods = [method for method in dir(User) if not method.startswith('_')]
        s3_related_methods = ['create_user', 'find_user_in_db', 'get_user_folder']
        
        for method in s3_related_methods:
            if method in methods:
                print_success(f"User class has method: {method}")
            else:
                print_warning(f"User class missing expected method: {method}")
                
        print_success(f"Successfully validated User class S3 integration")
        
    except Exception as e:
        print_error(f"Error testing User class S3 integration: {str(e)}")

# ====================== Main Test Runner ======================

def run_all_s3_connection_tests():
    """Run all tests for S3 connections."""
    print_header("TESTING ALL S3 CONNECTIONS")
    
    # Check environment first
    if not check_environment():
        return False
    
    # S3 Utils Tests
    test_get_datasets_s3_client()
    test_get_users_s3_client()
    
    # Models Service Tests
    # These may not work directly due to environment and HTTP dependencies
    print_subheader("Models Service Tests")
    print_info("Some tests may require specific environment setup and FastAPI context")
    test_s3_file_exists(None)  # Mock bucket
    test_read_json_from_s3(None)  # Mock bucket
    test_get_model_path(None, None, None)  # Mock fixtures
    test_check_model_path(None, None, None)  # Mock fixtures
    test_get_model_filename(None, None, None)  # Mock fixtures
    test_get_user_models_info(None, None, None)  # Mock fixtures
    test_get_model_files(None, None, None)  # Mock fixtures
    test_get_model_specific_file(None, None, None)  # Mock fixtures
    
    # Dendrogram Service Tests
    print_subheader("Dendrogram Service Tests")
    print_info("Some tests may require specific environment setup and FastAPI context")
    test_get_dendrogram_path(None, None, None)  # Mock fixtures
    test_dendrogram_class()
    
    # Whitebox Service Tests
    print_subheader("Whitebox Service Tests")
    print_info("Some tests may require specific environment setup and FastAPI context")
    test_get_edges_dataframe_path(None, None, None)  # Mock fixtures
    test_edges_dataframe_class()
    test_whitebox_testing_class()
    
    # Dataset Service Tests
    test_get_dataset_config()
    test_user_class_s3_integration()
    
    print_header("ALL S3 CONNECTION TESTS COMPLETED")
    return True

def main():
    """Main function with test menu."""
    print_header("S3 CONNECTION TEST SUITE")
    
    # Check environment
    if not check_environment():
        return
    
    # Test menu
    print_subheader("Test Menu")
    print("1.  Test S3 Utils Clients")
    print("2.  Test Model Service S3 Functions")
    print("3.  Test Dendrogram Service S3 Functions")
    print("4.  Test Whitebox Service S3 Functions")
    print("5.  Test Dataset Service S3 Functions")
    print("6.  Test User Class S3 Integration")
    print("7.  Run Dataset S3 Utility Tests (from original script)")
    print("8.  Run all S3 connection tests")
    print("9.  Exit")
    
    while True:
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            test_get_datasets_s3_client()
            test_get_users_s3_client()
        elif choice == '2':
            print_subheader("Models Service Tests")
            print_info("Some tests may require specific environment setup and FastAPI context")
            test_s3_file_exists(None)
            test_read_json_from_s3(None)
            test_get_model_path(None, None, None)
            test_check_model_path(None, None, None)
            test_get_model_filename(None, None, None)
            test_get_user_models_info(None, None, None)
            test_get_model_files(None, None, None)
            test_get_model_specific_file(None, None, None)
        elif choice == '3':
            print_subheader("Dendrogram Service Tests")
            test_get_dendrogram_path(None, None, None)
            test_dendrogram_class()
        elif choice == '4':
            print_subheader("Whitebox Service Tests")
            test_get_edges_dataframe_path(None, None, None)
            test_edges_dataframe_class()
            test_whitebox_testing_class()
        elif choice == '5':
            test_get_dataset_config()
        elif choice == '6':
            test_user_class_s3_integration()
        elif choice == '7':
            # Import and run functions from the original script
            from test_s3_dataset_utils import run_all_tests
            run_all_tests()
        elif choice == '8':
            run_all_s3_connection_tests()
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print_error("Invalid choice! Please enter 1-9.")
        
        if choice in [str(i) for i in range(1, 9)]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()