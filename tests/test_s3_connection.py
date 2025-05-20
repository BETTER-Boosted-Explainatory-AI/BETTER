import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from typing import Dict, Any, List, Tuple, Optional
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Import S3 handler and dataset loader
from utilss.s3_connector.s3_handler import S3Handler
from utilss.s3_connector.s3_dataset_loader import S3DatasetLoader
from utilss.enums.datasets_enum import DatasetsEnum

# For utilities module
from utilss.s3_connector.s3_dataset_utils import (
    load_dataset_numpy,
    load_cifar100_adversarial_or_clean,
    load_imagenet_adversarial_or_clean,
    get_dataset_config,
    load_dataset_folder,
    load_single_image,
    load_imagenet_train,
    load_cifar100_meta,
    load_dataset_split
)

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
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY', 
        'S3_BUCKET_NAME'
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

def visualize_numpy_data(data, title="Sample Data Visualization"):
    """Helper function to visualize NumPy data."""
    if not data:
        print_warning("No data to visualize")
        return
    
    if isinstance(data, tuple) and len(data) == 2:
        # Likely (images, labels) format
        images, labels = data
        if len(images) == 0:
            print_warning("No images to display")
            return
            
        # Display up to 5 random images
        fig, axes = plt.subplots(1, min(5, len(images)), figsize=(15, 3))
        if min(5, len(images)) == 1:
            axes = [axes]  # Make it iterable for consistent access
            
        for i in range(min(5, len(images))):
            idx = np.random.randint(0, len(images))
            ax = axes[i]
            img = images[idx]
            
            # Handle different image formats
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                
            ax.imshow(img)
            if labels is not None and idx < len(labels):
                ax.set_title(f"Label: {labels[idx]}")
            ax.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    elif isinstance(data, dict):
        # Dictionary of NumPy arrays
        if len(data) == 0:
            print_warning("Empty dictionary, nothing to display")
            return
            
        # Show up to 5 items
        keys = list(data.keys())[:5]
        fig, axes = plt.subplots(1, len(keys), figsize=(15, 3))
        if len(keys) == 1:
            axes = [axes]  # Make it iterable for consistent access
            
        for i, key in enumerate(keys):
            array = data[key]
            ax = axes[i]
            
            # Handle different array types
            if isinstance(array, np.ndarray):
                if len(array.shape) == 3:  # Image
                    if array.dtype != np.uint8:
                        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
                    
                    # Ensure correct channel ordering
                    if array.shape[0] == 3 and len(array.shape) == 3:  # [C,H,W] format
                        array = np.transpose(array, (1, 2, 0))
                        
                    ax.imshow(array)
                    ax.set_title(f"{key[:10]}...")
                else:
                    # Not an image, show first few values
                    ax.text(0.5, 0.5, f"Array shape: {array.shape}\nFirst values: {str(array.flatten()[:5])}", 
                           ha='center', va='center')
                    ax.set_title(f"{key[:10]}...")
            else:
                ax.text(0.5, 0.5, f"Type: {type(array).__name__}\nValue: {str(array)[:50]}", 
                       ha='center', va='center')
                ax.set_title(f"{key[:10]}...")
                
            ax.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    else:
        print_warning(f"Unsupported data type for visualization: {type(data)}")

def test_s3_handler():
    """Test all methods in the S3Handler class."""
    print_header("TESTING S3HANDLER CLASS")
    
    # Initialize S3Handler
    try:
        handler = S3Handler()
        print_success("Initialized S3Handler successfully")
    except Exception as e:
        print_error(f"Failed to initialize S3Handler: {str(e)}")
        return False
    
    # Test list_images
    print_subheader("Testing list_images()")
    try:
        # Test with CIFAR100 prefix
        cifar_objects = handler.list_images("cifar100/")
        if cifar_objects:
            print_success(f"Successfully listed {len(cifar_objects)} objects in CIFAR100")
            print_info(f"First 5 objects: {cifar_objects[:5]}")
        else:
            print_warning("No objects found with prefix 'cifar100/'")
        
        # Test with ImageNet prefix
        imagenet_objects = handler.list_images("imagenet/")
        if imagenet_objects:
            print_success(f"Successfully listed {len(imagenet_objects)} objects in ImageNet")
            print_info(f"First 5 objects: {imagenet_objects[:5]}")
        else:
            print_warning("No objects found with prefix 'imagenet/'")
    except Exception as e:
        print_error(f"Error in list_images(): {str(e)}")
    
    # Test get_folder_contents
    print_subheader("Testing get_folder_contents()")
    try:
        # Test with CIFAR100 clean folder
        cifar_clean = handler.get_folder_contents("cifar100", "clean")
        if cifar_clean:
            print_success(f"Successfully listed {len(cifar_clean)} objects in CIFAR100 clean folder")
            print_info(f"First 3 objects: {cifar_clean[:3]}")
        else:
            print_warning("No objects found in CIFAR100 clean folder")
        
        # Test with ImageNet clean folder
        imagenet_clean = handler.get_folder_contents("imagenet", "clean")
        if imagenet_clean:
            print_success(f"Successfully listed {len(imagenet_clean)} objects in ImageNet clean folder")
            print_info(f"First 3 objects: {imagenet_clean[:3]}")
        else:
            print_warning("No objects found in ImageNet clean folder")
    except Exception as e:
        print_error(f"Error in get_folder_contents(): {str(e)}")
    
    # Test download_file (with a small temporary file)
    print_subheader("Testing download_file()")
    temp_dir = tempfile.mkdtemp()
    try:
        # Try to download the first CIFAR100 file found
        cifar_files = handler.list_images("cifar100/")
        if cifar_files:
            test_file = cifar_files[0]
            local_path = os.path.join(temp_dir, os.path.basename(test_file))
            success = handler.download_file(test_file, local_path)
            
            if success and os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                print_success(f"Successfully downloaded {test_file} ({file_size} bytes)")
            else:
                print_error(f"Failed to download {test_file}")
        else:
            print_warning("No CIFAR100 files found to test download_file()")
    except Exception as e:
        print_error(f"Error in download_file(): {str(e)}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    # Test get_single_image
    print_subheader("Testing get_single_image()")
    try:
        # Try to get a single image from the first image file found
        image_files = []
        for prefix in ["cifar100/", "imagenet/"]:
            files = handler.list_images(prefix)
            image_files.extend([f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')])
        
        if image_files:
            test_image = image_files[0]
            image_data = handler.get_single_image(test_image)
            
            if image_data:
                print_success(f"Successfully retrieved image {test_image} ({len(image_data)} bytes)")
            else:
                print_error(f"Failed to retrieve image {test_image}")
        else:
            print_warning("No image files found to test get_single_image()")
    except Exception as e:
        print_error(f"Error in get_single_image(): {str(e)}")
    
    # Test get_numpy_data
    print_subheader("Testing get_numpy_data()")
    try:
        # Test with CIFAR100 clean folder
        cifar_numpy = handler.get_numpy_data("cifar100", "clean")
        if cifar_numpy:
            print_success(f"Successfully loaded {len(cifar_numpy)} NumPy arrays from CIFAR100 clean folder")
            print_info(f"Array keys: {list(cifar_numpy.keys())}")
            visualize_numpy_data(cifar_numpy, "CIFAR100 Clean Data")
        else:
            print_warning("No NumPy arrays found in CIFAR100 clean folder")
        
        # Test with ImageNet clean folder
        imagenet_numpy = handler.get_numpy_data("imagenet", "clean")
        if imagenet_numpy:
            print_success(f"Successfully loaded {len(imagenet_numpy)} NumPy arrays from ImageNet clean folder")
            print_info(f"Array keys: {list(imagenet_numpy.keys())}")
            visualize_numpy_data(imagenet_numpy, "ImageNet Clean Data")
        else:
            print_warning("No NumPy arrays found in ImageNet clean folder")
    except Exception as e:
        print_error(f"Error in get_numpy_data(): {str(e)}")
    
    # Test get_cifar100_as_numpy
    print_subheader("Testing get_cifar100_as_numpy()")
    try:
        # Test with CIFAR100 clean folder
        cifar_images, cifar_labels = handler.get_cifar100_as_numpy("clean")
        if cifar_images is not None and len(cifar_images) > 0:
            print_success(f"Successfully loaded CIFAR100 clean data as NumPy arrays")
            print_info(f"Images shape: {cifar_images.shape}, Labels shape: {cifar_labels.shape}")
            
            # Visualize the data
            print_info("Visualizing CIFAR100 Clean Data:")
            visualize_numpy_data((cifar_images, cifar_labels), "CIFAR100 Clean Data")
        else:
            print_warning("No CIFAR100 clean data found or data format is unexpected")
    except Exception as e:
        print_error(f"Error in get_cifar100_as_numpy(): {str(e)}")
    
    # Test get_cifar100_meta
    print_subheader("Testing get_cifar100_meta()")
    try:
        cifar_meta = handler.get_cifar100_meta()
        if cifar_meta:
            print_success(f"Successfully loaded CIFAR100 metadata")
            if isinstance(cifar_meta, dict):
                print_info(f"Metadata keys: {list(cifar_meta.keys())}")
            else:
                print_info(f"Metadata type: {type(cifar_meta)}")
        else:
            print_warning("No CIFAR100 metadata found")
    except Exception as e:
        print_error(f"Error in get_cifar100_meta(): {str(e)}")
    
    # Test load_python_module_from_s3
    print_subheader("Testing load_python_module_from_s3()")
    try:
        # Try to load CIFAR100 info module
        cifar_module = handler.load_python_module_from_s3("cifar100_info.py")
        if cifar_module:
            print_success(f"Successfully loaded Python module 'cifar100_info.py'")
            if hasattr(cifar_module, "CIFAR100_INFO"):
                cifar_info = getattr(cifar_module, "CIFAR100_INFO")
                print_info(f"CIFAR100_INFO available in module: {type(cifar_info)}")
                if isinstance(cifar_info, dict):
                    print_info(f"CIFAR100_INFO keys: {list(cifar_info.keys())}")
            else:
                print_warning("Module loaded but CIFAR100_INFO not found")
        else:
            print_warning("Failed to load Python module 'cifar100_info.py'")
    except Exception as e:
        print_error(f"Error in load_python_module_from_s3(): {str(e)}")
    
    return True

def test_s3_dataset_loader():
    """Test all methods in the S3DatasetLoader class."""
    print_header("TESTING S3DATASETLOADER CLASS")
    
    # Initialize S3DatasetLoader
    try:
        loader = S3DatasetLoader()
        print_success("Initialized S3DatasetLoader successfully")
    except Exception as e:
        print_error(f"Failed to initialize S3DatasetLoader: {str(e)}")
        return False
    
    # Test load_from_s3
    print_subheader("Testing load_from_s3()")
    try:
        # Test with CIFAR100
        print_info("Testing load_from_s3() with CIFAR100 dataset")
        print_warning("This operation may take a long time as it downloads the entire dataset")
        user_input = input("Do you want to proceed? (y/n): ").strip().lower()
        
        if user_input == 'y':
            temp_dir = loader.load_from_s3(DatasetsEnum.CIFAR100.value)
            try:
                print_success(f"Successfully loaded CIFAR100 dataset to {temp_dir}")
                print_info(f"Files in directory: {os.listdir(temp_dir)[:10]}...")
            finally:
                import shutil
                shutil.rmtree(temp_dir)
        else:
            print_info("Skipping full dataset download")
    except Exception as e:
        print_error(f"Error in load_from_s3(): {str(e)}")
    
    # Test load_folder
    print_subheader("Testing load_folder()")
    try:
        # Test with CIFAR100 clean folder
        temp_dir = loader.load_folder(DatasetsEnum.CIFAR100.value, "clean")
        try:
            print_success(f"Successfully loaded CIFAR100 clean folder to {temp_dir}")
            files = os.listdir(temp_dir)
            print_info(f"Files in directory: {files[:10]}...")
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print_error(f"Error in load_folder(): {str(e)}")
    
    # Test load_single_image
    print_subheader("Testing load_single_image()")
    try:
        # Find an image to test with
        s3_handler = S3Handler()
        image_files = []
        for prefix in ["cifar100/", "imagenet/"]:
            files = s3_handler.list_images(prefix)
            image_files.extend([f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')])
        
        if image_files:
            test_image = image_files[0]
            image_data = loader.load_single_image(test_image)
            
            if image_data:
                print_success(f"Successfully loaded image {test_image} ({len(image_data)} bytes)")
            else:
                print_error(f"Failed to load image {test_image}")
        else:
            print_warning("No image files found to test load_single_image()")
    except Exception as e:
        print_error(f"Error in load_single_image(): {str(e)}")
    
    # Test load_numpy_data
    print_subheader("Testing load_numpy_data()")
    try:
        # Test with CIFAR100 clean folder
        cifar_numpy = loader.load_numpy_data(DatasetsEnum.CIFAR100.value, "clean")
        if cifar_numpy:
            print_success(f"Successfully loaded {len(cifar_numpy)} NumPy arrays from CIFAR100 clean folder")
            print_info(f"Array keys: {list(cifar_numpy.keys())}")
            visualize_numpy_data(cifar_numpy, "CIFAR100 Clean NumPy Data")
        else:
            print_warning("No NumPy arrays found in CIFAR100 clean folder")
    except Exception as e:
        print_error(f"Error in load_numpy_data(): {str(e)}")
    
    # Test load_cifar100_numpy
    print_subheader("Testing load_cifar100_numpy()")
    try:
        # Test with CIFAR100 clean folder
        cifar_images, cifar_labels = loader.load_cifar100_numpy("clean")
        if cifar_images is not None and len(cifar_images) > 0:
            print_success(f"Successfully loaded CIFAR100 clean data as NumPy arrays")
            print_info(f"Images shape: {cifar_images.shape}, Labels shape: {cifar_labels.shape}")
            
            # Visualize the data
            print_info("Visualizing CIFAR100 Clean Data:")
            visualize_numpy_data((cifar_images, cifar_labels), "CIFAR100 Clean Data")
        else:
            print_warning("No CIFAR100 clean data found or data format is unexpected")
    except Exception as e:
        print_error(f"Error in load_cifar100_numpy(): {str(e)}")
    
    # Test load_cifar100_meta
    print_subheader("Testing load_cifar100_meta()")
    try:
        cifar_meta = loader.load_cifar100_meta()
        if cifar_meta:
            print_success(f"Successfully loaded CIFAR100 metadata")
            if isinstance(cifar_meta, dict):
                print_info(f"Metadata keys: {list(cifar_meta.keys())}")
            else:
                print_info(f"Metadata type: {type(cifar_meta)}")
        else:
            print_warning("No CIFAR100 metadata found")
    except Exception as e:
        print_error(f"Error in load_cifar100_meta(): {str(e)}")
    
    # Test get_dataset_info
    print_subheader("Testing get_dataset_info()")
    try:
        # Test with CIFAR100
        cifar_info = loader.get_dataset_info(DatasetsEnum.CIFAR100.value)
        if cifar_info:
            print_success(f"Successfully loaded CIFAR100 dataset info")
            print_info(f"Info keys: {list(cifar_info.keys())}")
        else:
            print_warning("No CIFAR100 dataset info found")
        
        # Test with ImageNet
        imagenet_info = loader.get_dataset_info(DatasetsEnum.IMAGENET.value)
        if imagenet_info:
            print_success(f"Successfully loaded ImageNet dataset info")
            print_info(f"Info keys: {list(imagenet_info.keys())}")
        else:
            print_warning("No ImageNet dataset info found")
    except Exception as e:
        print_error(f"Error in get_dataset_info(): {str(e)}")
    
    # Test load_dataset_split
    print_subheader("Testing load_dataset_split()")
    try:
        # Test with CIFAR100 test split
        print_info("Testing load_dataset_split() with CIFAR100 test split")
        temp_dir = loader.load_dataset_split(DatasetsEnum.CIFAR100.value, "test")
        try:
            print_success(f"Successfully loaded CIFAR100 test split to {temp_dir}")
            files = os.listdir(temp_dir)
            print_info(f"Files in directory: {files[:10]}...")
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print_error(f"Error in load_dataset_split(): {str(e)}")
    
    return True

def test_utility_functions():
    """Test all utility functions from s3_dataset_utils.py"""
    print_header("TESTING UTILITY FUNCTIONS")
    
    # Test get_dataset_config
    print_subheader("Testing get_dataset_config()")
    try:
        # Test with CIFAR100
        cifar_config = get_dataset_config('cifar100')
        if cifar_config:
            print_success(f"Successfully loaded CIFAR100 config")
            print_info(f"Config keys: {list(cifar_config.keys())}")
        else:
            print_warning("No CIFAR100 config found")
    except Exception as e:
        print_error(f"Error in get_dataset_config(): {str(e)}")
    
    # Test load_dataset_numpy
    print_subheader("Testing load_dataset_numpy()")
    try:
        # Test with CIFAR100 clean folder
        cifar_numpy = load_dataset_numpy('cifar100', 'clean')
        if cifar_numpy:
            print_success(f"Successfully loaded {len(cifar_numpy)} NumPy arrays from CIFAR100 clean folder")
            print_info(f"Array keys: {list(cifar_numpy.keys())}")
            visualize_numpy_data(cifar_numpy, "CIFAR100 Clean NumPy Data")
        else:
            print_warning("No NumPy arrays found in CIFAR100 clean folder")
    except Exception as e:
        print_error(f"Error in load_dataset_numpy(): {str(e)}")
    
    # Test load_cifar100_adversarial_or_clean
    print_subheader("Testing load_cifar100_adversarial_or_clean()")
    try:
        # Test with clean folder
        cifar_clean = load_cifar100_adversarial_or_clean('clean')
        if cifar_clean:
            print_success(f"Successfully loaded {len(cifar_clean)} NumPy arrays from CIFAR100 clean folder")
            print_info(f"Array keys: {list(cifar_clean.keys())}")
            visualize_numpy_data(cifar_clean, "CIFAR100 Clean Data")
        else:
            print_warning("No NumPy arrays found in CIFAR100 clean folder")
    except Exception as e:
        print_error(f"Error in load_cifar100_adversarial_or_clean(): {str(e)}")
    
    # Test load_imagenet_adversarial_or_clean
    print_subheader("Testing load_imagenet_adversarial_or_clean()")
    try:
        # Test with clean folder
        imagenet_clean = load_imagenet_adversarial_or_clean('clean')
        if imagenet_clean:
            print_success(f"Successfully loaded {len(imagenet_clean)} NumPy arrays from ImageNet clean folder")
            print_info(f"Array keys: {list(imagenet_clean.keys())}")
            visualize_numpy_data(imagenet_clean, "ImageNet Clean Data")
        else:
            print_warning("No NumPy arrays found in ImageNet clean folder")
    except Exception as e:
        print_error(f"Error in load_imagenet_adversarial_or_clean(): {str(e)}")
    
    # Test load_dataset_folder
    print_subheader("Testing load_dataset_folder()")
    try:
        # Test with CIFAR100 clean folder
        temp_dir = load_dataset_folder('cifar100', 'clean')
        try:
            print_success(f"Successfully loaded CIFAR100 clean folder to {temp_dir}")
            files = os.listdir(temp_dir)
            print_info(f"Files in directory: {files[:10]}...")
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print_error(f"Error in load_dataset_folder(): {str(e)}")
    
    # Test load_cifar100_meta
    print_subheader("Testing load_cifar100_meta()")
    try:
        cifar_meta = load_cifar100_meta()
        if cifar_meta:
            print_success(f"Successfully loaded CIFAR100 metadata")
            if isinstance(cifar_meta, dict):
                print_info(f"Metadata keys: {list(cifar_meta.keys())}")
            else:
                print_info(f"Metadata type: {type(cifar_meta)}")
        else:
            print_warning("No CIFAR100 metadata found")
    except Exception as e:
        print_error(f"Error in load_cifar100_meta(): {str(e)}")
    
    # Test load_dataset_split
    print_subheader("Testing load_dataset_split()")
    try:
        # Test with CIFAR100 test split
        temp_dir = load_dataset_split('cifar100', 'test')
        try:
            print_success(f"Successfully loaded CIFAR100 test split to {temp_dir}")
            files = os.listdir(temp_dir)
            print_info(f"Files in directory: {files[:10]}...")
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print_error(f"Error in load_dataset_split(): {str(e)}")
    
    return True

def main():
    """Main function to run all tests."""
    print_header("S3 DATASET SERVICES TESTING SUITE")
    
    # Check environment
    if not check_environment():
        return
    
    # Test menu
    print_subheader("Test Menu")
    print("1. Test S3Handler only")
    print("2. Test S3DatasetLoader only")
    print("3. Test Utility Functions only")
    print("4. Run all tests")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        test_s3_handler()
    elif choice == '2':
        test_s3_dataset_loader()
    elif choice == '3':
        test_utility_functions()
    elif choice == '4':
        test_s3_handler()
        test_s3_dataset_loader()
        test_utility_functions()
    elif choice == '5':
        print("Exiting...")
        return
    else:
        print_error("Invalid choice!")
    
    print_header("TESTING COMPLETED")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user!")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()