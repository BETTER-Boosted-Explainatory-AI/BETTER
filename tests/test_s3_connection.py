import os
import sys
import numpy as np
import logging
import boto3
import json
from botocore.exceptions import ClientError

# Import path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Import services to test
from services.models_service import (
    s3_file_exists, read_json_from_s3, _get_model_path, _get_model_filename,
    _check_model_path, get_user_models_info
)
from services.dendrogram_service import _get_dendrogram_path, _get_sub_dendrogram
from services.whitebox_testing_service import _get_edges_dataframe_path
from services.dataset_service import _get_dataset_config, get_dataset_labels, _load_dataset
from utilss.s3_connector.s3_dataset_utils import (
    load_dataset_numpy, load_cifar100_adversarial_or_clean,
    load_imagenet_adversarial_or_clean, load_dataset_folder,
    get_image_stream, load_cifar100_as_numpy
)
from utilss.s3_utils import get_users_s3_client, get_datasets_s3_client
from utilss.classes.user import User
from utilss.classes.datasets.cifar100 import Cifar100
from utilss.classes.datasets.imagenet import ImageNet

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# ====================== S3 Client Tests ======================

def test_s3_clients():
    """Test S3 client connections."""
    print_subheader("Testing S3 Clients")
    
    # Test datasets S3 client
    try:
        datasets_client = get_datasets_s3_client()
        datasets_bucket = os.environ.get('S3_DATASETS_BUCKET_NAME')
        response = datasets_client.list_objects_v2(Bucket=datasets_bucket, MaxKeys=1)
        if 'Contents' in response:
            print_success(f"Successfully connected to datasets S3 bucket: {datasets_bucket}")
            print_info(f"Found {len(response['Contents'])} objects")
        else:
            print_warning(f"Connected to datasets S3 bucket: {datasets_bucket} but it appears empty")
    except Exception as e:
        print_error(f"Failed to connect to datasets S3 bucket: {str(e)}")
    
    # Test users S3 client
    try:
        users_client = get_users_s3_client()
        users_bucket = os.environ.get('S3_USERS_BUCKET_NAME')
        response = users_client.list_objects_v2(Bucket=users_bucket, MaxKeys=1)
        if 'Contents' in response:
            print_success(f"Successfully connected to users S3 bucket: {users_bucket}")
            print_info(f"Found {len(response['Contents'])} objects")
        else:
            print_warning(f"Connected to users S3 bucket: {users_bucket} but it appears empty")
    except Exception as e:
        print_error(f"Failed to connect to users S3 bucket: {str(e)}")

# ====================== User Service Tests ======================

def test_user_operations():
    """Test User class S3 operations without modifying data."""
    print_subheader("Testing User Class S3 Operations")
    
    # List users and pick one for testing
    try:
        users_client = get_users_s3_client()
        users_bucket = os.environ.get('S3_USERS_BUCKET_NAME')
        
        # Try to read users.json
        try:
            response = users_client.get_object(Bucket=users_bucket, Key="users.json")
            users = json.loads(response['Body'].read().decode('utf-8'))
            if users and len(users) > 0:
                print_success(f"Successfully read users.json, found {len(users)} users")
                
                # Pick first user for testing
                test_user_id = users[0]["id"]
                test_user_email = users[0]["email"]
                print_info(f"Using user: {test_user_id} with email: {test_user_email} for testing")
                
                # Test User class functionality
                user = User(user_id=test_user_id, email=test_user_email)
                
                # Test finding user in DB
                found_user = user.find_user_in_db()
                if found_user:
                    print_success(f"Successfully found user in DB: {test_user_id}")
                else:
                    print_warning(f"Could not find user in DB: {test_user_id}")
                
                # Test loading models
                user.load_models()
                models = user.get_models()
                print_success(f"Successfully loaded models for user, found {len(models) if isinstance(models, list) else 0} models")
                
                # Test loading current model (if exists)
                try:
                    current_model = user.get_current_model()
                    if current_model and isinstance(current_model, dict):
                        print_success(f"Successfully loaded current model: {current_model.get('model_id', 'Unknown ID')}")
                    else:
                        print_info("No current model set for user")
                except Exception as e:
                    print_warning(f"Error loading current model: {str(e)}")
                
                return test_user_id, test_user_email, models
            else:
                print_warning("users.json exists but contains no users")
        except users_client.exceptions.NoSuchKey:
            print_warning("users.json not found in S3 bucket")
        except Exception as e:
            print_error(f"Error reading users.json: {str(e)}")
    except Exception as e:
        print_error(f"Error testing User class: {str(e)}")
    
    return None, None, []

# ====================== Models Service Tests ======================

def test_models_service(user_id, models):
    """Test models_service functions with actual data."""
    print_subheader("Testing Models Service Functions")
    
    if not user_id or not models or len(models) == 0:
        print_warning("No user ID or models available for testing")
        return None
    
    # Get the first model for testing
    test_model = models[0] if isinstance(models, list) and len(models) > 0 else None
    if not test_model:
        print_warning("No models available for testing")
        return None
    
    model_id = test_model.get("model_id")
    print_info(f"Using model: {model_id} for testing")
    
    # Test _get_model_path
    try:
        model_path = _get_model_path(user_id, model_id)
        if model_path:
            print_success(f"Successfully got model path: {model_path}")
        else:
            print_warning(f"Could not get model path for user {user_id}, model {model_id}")
    except Exception as e:
        print_error(f"Error in _get_model_path(): {str(e)}")
    
    # Test _check_model_path with a graph type
    try:
        graph_types = ["activation", "weights"]
        for graph_type in graph_types:
            try:
                model_path = _check_model_path(user_id, model_id, graph_type)
                if model_path:
                    print_success(f"Successfully checked model path with graph type {graph_type}: {model_path}")
                    
                    # Now we have a valid graph type, use it for further tests
                    
                    # Test _get_model_filename
                    try:
                        model_filename = _get_model_filename(user_id, model_id, graph_type)
                        if model_filename:
                            print_success(f"Successfully got model filename: {model_filename}")
                        else:
                            print_warning(f"Could not get model filename for graph type {graph_type}")
                    except Exception as e:
                        print_error(f"Error in _get_model_filename(): {str(e)}")
                    
                    # Test get_user_models_info with a mock user
                    try:
                        class MockUser:
                            def get_user_folder(self):
                                return user_id
                                
                            def __init__(self, user_id=user_id, email=None):
                                self.user_id = user_id
                                self.email = email
                        
                        model_info = get_user_models_info(MockUser(), model_id)
                        if model_info:
                            print_success(f"Successfully got model info: {model_info.get('model_id')}")
                        else:
                            print_warning(f"Could not get model info for model {model_id}")
                    except Exception as e:
                        print_error(f"Error in get_user_models_info(): {str(e)}")
                    
                    return model_id, graph_type
                else:
                    print_info(f"Model path not found for graph type {graph_type}")
            except Exception as e:
                print_info(f"Graph type {graph_type} not available: {str(e)}")
    except Exception as e:
        print_error(f"Error testing _check_model_path(): {str(e)}")
    
    return model_id, None

# ====================== Dendrogram Service Tests ======================

def test_dendrogram_service(user_id, model_id, graph_type):
    """Test dendrogram_service functions with actual data."""
    print_subheader("Testing Dendrogram Service Functions")
    
    if not user_id or not model_id or not graph_type:
        print_warning("Missing user ID, model ID, or graph type for dendrogram tests")
        return
    
    # Test _get_dendrogram_path
    try:
        dendrogram_path = _get_dendrogram_path(user_id, model_id, graph_type)
        if dendrogram_path:
            print_success(f"Successfully got dendrogram path: {dendrogram_path}")
            
            # Now we can test other dendrogram functions with a mock user
            try:
                class MockUser:
                    def get_user_id(self):
                        return user_id
                        
                    def get_user_folder(self):
                        return user_id
                        
                    def __init__(self, user_id=user_id, email=None):
                        self.user_id = user_id
                        self.email = email
                
                # Test _get_sub_dendrogram with no selected labels (uses defaults)
                try:
                    sub_dendrogram, selected_labels = _get_sub_dendrogram(MockUser(), model_id, graph_type, None)
                    if sub_dendrogram:
                        print_success(f"Successfully got sub dendrogram with {len(selected_labels)} default selected labels")
                    else:
                        print_warning("Could not get sub dendrogram with default labels")
                except Exception as e:
                    print_error(f"Error in _get_sub_dendrogram(): {str(e)}")
                
                # Test _get_common_ancestor_subtree with selected labels
                if selected_labels and len(selected_labels) >= 2:
                    try:
                        # Take first two labels for testing
                        test_labels = selected_labels[:2]
                        subtree, labels = _get_common_ancestor_subtree(MockUser(), model_id, graph_type, test_labels)
                        if subtree:
                            print_success(f"Successfully got common ancestor subtree for labels: {test_labels}")
                        else:
                            print_warning(f"Could not get common ancestor subtree for labels: {test_labels}")
                    except Exception as e:
                        print_error(f"Error in _get_common_ancestor_subtree(): {str(e)}")
            except Exception as e:
                print_error(f"Error testing dendrogram functions with mock user: {str(e)}")
        else:
            print_warning(f"Could not get dendrogram path for user {user_id}, model {model_id}, graph type {graph_type}")
    except Exception as e:
        print_error(f"Error in _get_dendrogram_path(): {str(e)}")

# ====================== Whitebox Service Tests ======================

def test_whitebox_service(user_id, model_id, graph_type):
    """Test whitebox_service functions with actual data."""
    print_subheader("Testing Whitebox Service Functions")
    
    if not user_id or not model_id or not graph_type:
        print_warning("Missing user ID, model ID, or graph type for whitebox tests")
        return
    
    # Test _get_edges_dataframe_path
    try:
        edges_df_path = _get_edges_dataframe_path(user_id, model_id, graph_type)
        if edges_df_path:
            print_success(f"Successfully got edges dataframe path: {edges_df_path}")
        else:
            print_warning(f"Could not get edges dataframe path for user {user_id}, model {model_id}, graph type {graph_type}")
    except Exception as e:
        print_error(f"Error in _get_edges_dataframe_path(): {str(e)}")

# ====================== Dataset Service Tests ======================

def test_dataset_service():
    """Test dataset_service functions with actual data."""
    print_subheader("Testing Dataset Service Functions")
    
    # Test _get_dataset_config for CIFAR100
    try:
        cifar_config = _get_dataset_config('cifar100')
        if cifar_config:
            print_success("Successfully got CIFAR100 dataset config")
            print_info(f"Config keys: {list(cifar_config.keys())}")
            
            # Test get_dataset_labels
            try:
                labels = get_dataset_labels('cifar100')
                if labels:
                    print_success(f"Successfully got CIFAR100 labels: {len(labels)} labels")
                else:
                    print_warning("Could not get CIFAR100 labels")
            except Exception as e:
                print_error(f"Error in get_dataset_labels(): {str(e)}")
        else:
            print_warning("Could not get CIFAR100 dataset config")
    except Exception as e:
        print_error(f"Error in _get_dataset_config(): {str(e)}")
    
    # Test _get_dataset_config for ImageNet
    try:
        imagenet_config = _get_dataset_config('imagenet')
        if imagenet_config:
            print_success("Successfully got ImageNet dataset config")
            print_info(f"Config keys: {list(imagenet_config.keys())}")
        else:
            print_warning("Could not get ImageNet dataset config")
    except Exception as e:
        print_warning(f"Error in _get_dataset_config() for ImageNet: {str(e)}")

# ====================== Dataset Classes Tests ======================

def test_dataset_classes():
    """Test dataset classes directly."""
    print_subheader("Testing Dataset Classes")
    
    # Test Cifar100 class
    try:
        # Create Cifar100 dataset
        cifar_dataset = Cifar100()
        
        # Test loading from S3
        try:
            # Get S3 client
            s3_client = get_datasets_s3_client()
            bucket = os.environ.get('S3_DATASETS_BUCKET_NAME')
            
            # Test load_from_s3 method
            x_train, y_train = cifar_dataset.load_from_s3(s3_client, bucket, 'cifar100/')
            if x_train is not None and len(x_train) > 0:
                print_success(f"Successfully loaded CIFAR100 train data using dataset class: {len(x_train)} images")
            else:
                print_warning("Could not load CIFAR100 train data using dataset class")
        except Exception as e:
            print_error(f"Error in Cifar100.load_from_s3(): {str(e)}")
        
        # Test load method
        try:
            result = cifar_dataset.load('cifar100')
            if result:
                print_success(f"Successfully loaded CIFAR100 dataset: {len(cifar_dataset.x_train)} train images, {len(cifar_dataset.x_test)} test images")
            else:
                print_warning("Could not load CIFAR100 dataset")
        except Exception as e:
            print_error(f"Error in Cifar100.load(): {str(e)}")
        
    except Exception as e:
        print_error(f"Error testing Cifar100 class: {str(e)}")
    
    # Test ImageNet class if available
    try:
        # Create ImageNet dataset
        imagenet_dataset = ImageNet()
        
        # Test load method if available
        if hasattr(imagenet_dataset, 'load'):
            try:
                result = imagenet_dataset.load('imagenet')
                if result:
                    print_success("Successfully loaded ImageNet dataset")
                else:
                    print_warning("Could not load ImageNet dataset")
            except Exception as e:
                print_warning(f"Error in ImageNet.load(): {str(e)}")
        else:
            print_info("ImageNet class does not have a load method")
        
        # Test load_from_s3 method if available
        if hasattr(imagenet_dataset, 'load_from_s3'):
            try:
                s3_client = get_datasets_s3_client()
                bucket = os.environ.get('S3_DATASETS_BUCKET_NAME')
                result = imagenet_dataset.load_from_s3(s3_client, bucket, 'imagenet/')
                if result:
                    print_success("Successfully loaded ImageNet data using dataset class")
                else:
                    print_warning("Could not load ImageNet data using dataset class")
            except Exception as e:
                print_warning(f"Error in ImageNet.load_from_s3(): {str(e)}")
        else:
            print_info("ImageNet class does not have a load_from_s3 method")
    except Exception as e:
        print_warning(f"Error testing ImageNet class: {str(e)}")

# ====================== Direct Load Dataset Split Test ======================

def test_direct_load_dataset_split():
    """Test loading dataset splits directly with dataset classes."""
    print_subheader("Testing Direct Dataset Split Loading")
    
    # Test CIFAR100 train split using S3 client directly
    try:
        s3_client = get_datasets_s3_client()
        bucket = os.environ.get('S3_DATASETS_BUCKET_NAME')
        
        # List objects with CIFAR100 train prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='cifar100/train')
        
        if 'Contents' in response:
            print_success(f"Successfully listed CIFAR100 train objects: {len(response['Contents'])} objects")
            
            # Show a few objects
            if len(response['Contents']) > 0:
                sample_keys = [obj['Key'] for obj in response['Contents'][:3]]
                print_info(f"Sample keys: {sample_keys}")
        else:
            print_warning("No CIFAR100 train objects found")
    except Exception as e:
        print_error(f"Error listing CIFAR100 train objects: {str(e)}")
    
    # Try loading using unpickle_from_s3
    try:
        from utilss.s3_connector.s3_dataset_utils import unpickle_from_s3
        
        bucket = os.environ.get('S3_DATASETS_BUCKET_NAME')
        data = unpickle_from_s3(bucket, 'cifar100/train')
        
        if data:
            print_success("Successfully loaded CIFAR100 train data using unpickle_from_s3")
            if isinstance(data, dict):
                print_info(f"Data keys: {[k.decode('utf-8') if isinstance(k, bytes) else k for k in data.keys()]}")
                if b'data' in data:
                    print_info(f"Data shape: {data[b'data'].shape}")
        else:
            print_warning("Could not load CIFAR100 train data using unpickle_from_s3")
    except Exception as e:
        print_error(f"Error in unpickle_from_s3(): {str(e)}")

# ====================== S3 Dataset Utils Tests ======================

def test_s3_dataset_utils():
    """Test S3 dataset utility functions."""
    print_subheader("Testing S3 Dataset Utility Functions")
    
    # Test load_dataset_numpy
    try:
        cifar_data = load_dataset_numpy('cifar100', 'clean')
        if cifar_data:
            print_success(f"Successfully loaded CIFAR100 clean numpy data: {len(cifar_data)} arrays")
        else:
            print_warning("Could not load CIFAR100 clean numpy data")
    except Exception as e:
        print_error(f"Error in load_dataset_numpy(): {str(e)}")
    
    # Test load_cifar100_adversarial_or_clean
    try:
        cifar_clean = load_cifar100_adversarial_or_clean('clean')
        if cifar_clean:
            print_success(f"Successfully loaded CIFAR100 clean data: {len(cifar_clean)} arrays")
        else:
            print_warning("Could not load CIFAR100 clean data")
    except Exception as e:
        print_error(f"Error in load_cifar100_adversarial_or_clean(): {str(e)}")
    
    # Test load_dataset_folder
    try:
        cifar_files = load_dataset_folder('cifar100', 'clean')
        if cifar_files:
            print_success(f"Successfully listed CIFAR100 clean folder: {len(cifar_files)} files")
        else:
            print_warning("Could not list CIFAR100 clean folder")
    except Exception as e:
        print_error(f"Error in load_dataset_folder(): {str(e)}")
    
    # Test load_cifar100_as_numpy
    try:
        images, labels = load_cifar100_as_numpy('clean')
        if images is not None and len(images) > 0:
            print_success(f"Successfully loaded CIFAR100 clean as numpy: {len(images)} images")
        else:
            print_warning("Could not load CIFAR100 clean as numpy")
    except Exception as e:
        print_error(f"Error in load_cifar100_as_numpy(): {str(e)}")
    

    try:
        cifar_dataset = _load_dataset('cifar100')
        if cifar_dataset and hasattr(cifar_dataset, 'x_train') and cifar_dataset.x_train is not None:
            print_success(f"Successfully loaded CIFAR100 train split using _load_dataset: {len(cifar_dataset.x_train)} images")
        else:
            print_warning("Could not load CIFAR100 train split using _load_dataset")
    except Exception as e:
        print_error(f"Error in _load_dataset(): {str(e)}")
        


# ====================== Main Test Runner ======================

def run_all_s3_connection_tests():
    """Run all tests for S3 connections."""
    print_header("TESTING ALL S3 CONNECTIONS")
    
    # Check environment first
    if not check_environment():
        return False
    
    # Test S3 clients
    test_s3_clients()
    
    # Test User operations and get test user/model
    user_id, user_email, models = test_user_operations()
    
    # If we have a user and models, test other services
    if user_id and models:
        # Test Models Service
        model_id, graph_type = test_models_service(user_id, models)
        
        # If we have a model and graph type, test dendrogram and whitebox services
        if model_id and graph_type:
            test_dendrogram_service(user_id, model_id, graph_type)
            test_whitebox_service(user_id, model_id, graph_type)
    
    # Test Dataset Service
    test_dataset_service()
    
    # Test Dataset Classes directly
    test_dataset_classes()
    
    # Test direct dataset split loading
    test_direct_load_dataset_split()
    
    # Test S3 Dataset Utils
    test_s3_dataset_utils()
    
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
    print("1. Test S3 Clients")
    print("2. Test User Operations")
    print("3. Test Models Service")
    print("4. Test Dendrogram Service")
    print("5. Test Whitebox Service")
    print("6. Test Dataset Service")
    print("7. Test Dataset Classes")
    print("8. Test Direct Dataset Split Loading")
    print("9. Test S3 Dataset Utils")
    print("10. Run All S3 Connection Tests")
    print("11. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-11): ").strip()
        
        if choice == '1':
            test_s3_clients()
        elif choice == '2':
            user_id, user_email, models = test_user_operations()
            if not user_id:
                print_warning("No user available for testing subsequent services")
        elif choice == '3':
            # First get a test user
            user_id, user_email, models = test_user_operations()
            if user_id and models:
                test_models_service(user_id, models)
            else:
                print_warning("No user available for testing Models Service")
        elif choice == '4':
            # First get a test user and model
            user_id, user_email, models = test_user_operations()
            if user_id and models:
                model_id, graph_type = test_models_service(user_id, models)
                if model_id and graph_type:
                    test_dendrogram_service(user_id, model_id, graph_type)
                else:
                    print_warning("No model or graph type available for testing Dendrogram Service")
            else:
                print_warning("No user available for testing Dendrogram Service")
        elif choice == '5':
            # First get a test user and model
            user_id, user_email, models = test_user_operations()
            if user_id and models:
                model_id, graph_type = test_models_service(user_id, models)
                if model_id and graph_type:
                    test_whitebox_service(user_id, model_id, graph_type)
                else:
                    print_warning("No model or graph type available for testing Whitebox Service")
            else:
                print_warning("No user available for testing Whitebox Service")
        elif choice == '6':
            test_dataset_service()
        elif choice == '7':
            test_dataset_classes()
        elif choice == '8':
            test_direct_load_dataset_split()
        elif choice == '9':
            test_s3_dataset_utils()
        elif choice == '10':
            run_all_s3_connection_tests()
        elif choice == '11':
            print("Exiting...")
            break
        else:
            print_error("Invalid choice! Please enter 1-11.")
        
        if choice in [str(i) for i in range(1, 11)]:
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