import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Import all functions from s3_dataset_utils
from utilss.s3_connector.s3_dataset_utils import (
    load_dataset_numpy,
    load_cifar100_adversarial_or_clean,
    load_imagenet_adversarial_or_clean,
    get_dataset_config,
    load_dataset_folder,
    load_single_image,
    get_image_stream,
    load_imagenet_train,
    load_cifar100_as_numpy,
    load_cifar100_meta,
    load_dataset_split,
    unpickle_from_s3
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
    if data is None:
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
            axes = [axes]
            
        for i in range(min(5, len(images))):
            idx = np.random.randint(0, len(images))
            ax = axes[i]
            img = images[idx]
            
            # Handle different image formats
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            # Check if channels need to be transposed
            if len(img.shape) == 3 and img.shape[0] == 3 and img.shape[2] != 3:
                img = np.transpose(img, (1, 2, 0))
                
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
            axes = [axes]
            
        for i, key in enumerate(keys):
            array = data[key]
            ax = axes[i]
            
            # Handle different array types
            if isinstance(array, np.ndarray):
                if len(array.shape) >= 3:  # Image
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

def test_load_dataset_numpy():
    """Test load_dataset_numpy function."""
    print_subheader("Testing load_dataset_numpy()")
    try:
        # Test with CIFAR100 clean
        cifar_numpy = load_dataset_numpy('cifar100', 'clean')
        if cifar_numpy:
            print_success(f"Successfully loaded {len(cifar_numpy)} NumPy arrays from CIFAR100 clean")
            print_info(f"Array keys: {list(cifar_numpy.keys())[:5]}")
            visualize_numpy_data(cifar_numpy, "CIFAR100 Clean NumPy Data")
        else:
            print_warning("No NumPy arrays found in CIFAR100 clean folder")
            
        # Test with ImageNet clean
        imagenet_numpy = load_dataset_numpy('imagenet', 'clean')
        if imagenet_numpy:
            print_success(f"Successfully loaded {len(imagenet_numpy)} NumPy arrays from ImageNet clean")
            print_info(f"Array keys: {list(imagenet_numpy.keys())[:5]}")
        else:
            print_warning("No NumPy arrays found in ImageNet clean folder")
            
    except Exception as e:
        print_error(f"Error in load_dataset_numpy(): {str(e)}")

def test_load_cifar100_adversarial_or_clean():
    """Test load_cifar100_adversarial_or_clean function."""
    print_subheader("Testing load_cifar100_adversarial_or_clean()")
    try:
        # Test with clean folder
        cifar_clean = load_cifar100_adversarial_or_clean('clean')
        if cifar_clean:
            print_success(f"Successfully loaded {len(cifar_clean)} arrays from CIFAR100 clean")
            print_info(f"Array keys: {list(cifar_clean.keys())[:5]}")
            visualize_numpy_data(cifar_clean, "CIFAR100 Clean Data")
        else:
            print_warning("No data found in CIFAR100 clean folder")
            
        # Test with adversarial folder if exists
        try:
            cifar_adv = load_cifar100_adversarial_or_clean('adversarial')
            if cifar_adv:
                print_success(f"Successfully loaded {len(cifar_adv)} arrays from CIFAR100 adversarial")
                print_info(f"Array keys: {list(cifar_adv.keys())[:5]}")
        except:
            print_info("CIFAR100 adversarial folder not available")
            
    except Exception as e:
        print_error(f"Error in load_cifar100_adversarial_or_clean(): {str(e)}")

def test_load_imagenet_adversarial_or_clean():
    """Test load_imagenet_adversarial_or_clean function."""
    print_subheader("Testing load_imagenet_adversarial_or_clean()")
    try:
        # Test with clean folder
        imagenet_clean = load_imagenet_adversarial_or_clean('clean')
        if imagenet_clean:
            print_success(f"Successfully loaded {len(imagenet_clean)} arrays from ImageNet clean")
            print_info(f"Array keys: {list(imagenet_clean.keys())[:5]}")
            visualize_numpy_data(imagenet_clean, "ImageNet Clean Data")
        else:
            print_warning("No data found in ImageNet clean folder")
            
    except Exception as e:
        print_error(f"Error in load_imagenet_adversarial_or_clean(): {str(e)}")

def test_get_dataset_config():
    """Test get_dataset_config function."""
    print_subheader("Testing get_dataset_config()")
    try:
        # Test with CIFAR100
        cifar_config = get_dataset_config('cifar100')
        if cifar_config:
            print_success("Successfully loaded CIFAR100 config")
            print_info(f"Config keys: {list(cifar_config.keys())}")
            if 'labels' in cifar_config:
                print_info(f"Number of labels: {len(cifar_config['labels'])}")
        else:
            print_warning("No CIFAR100 config found")
            
        # Test with ImageNet
        imagenet_config = get_dataset_config('imagenet')
        if imagenet_config:
            print_success("Successfully loaded ImageNet config")
            print_info(f"Config keys: {list(imagenet_config.keys())}")
        else:
            print_warning("No ImageNet config found")
            
    except Exception as e:
        print_error(f"Error in get_dataset_config(): {str(e)}")

def test_load_dataset_folder():
    """Test load_dataset_folder function."""
    print_subheader("Testing load_dataset_folder()")
    try:
        # Test with CIFAR100 clean
        cifar_files = load_dataset_folder('cifar100', 'clean')
        if cifar_files:
            print_success(f"Successfully listed {len(cifar_files)} files in CIFAR100 clean")
            print_info(f"First 5 files: {cifar_files[:5]}")
        else:
            print_warning("No files found in CIFAR100 clean folder")
            
        # Test with ImageNet clean
        imagenet_files = load_dataset_folder('imagenet', 'clean')
        if imagenet_files:
            print_success(f"Successfully listed {len(imagenet_files)} files in ImageNet clean")
            print_info(f"First 5 files: {imagenet_files[:5]}")
        else:
            print_warning("No files found in ImageNet clean folder")
            
    except Exception as e:
        print_error(f"Error in load_dataset_folder(): {str(e)}")

def test_load_single_image():
    """Test load_single_image function."""
    print_subheader("Testing load_single_image()")
    try:
        # First get a file to test with
        files = load_dataset_folder('cifar100', 'clean')
        if files:
            # Find a .npy file
            npy_files = [f for f in files if f.endswith('.npy')]
            if npy_files:
                test_file = npy_files[0]
                image_data = load_single_image(test_file)
                
                if image_data:
                    print_success(f"Successfully loaded {test_file} ({len(image_data)} bytes)")
                    
                    # Try to load as numpy array
                    try:
                        numpy_array = np.load(io.BytesIO(image_data), allow_pickle=True)
                        print_info(f"Loaded NumPy array with shape: {numpy_array.shape}")
                    except Exception as e:
                        print_warning(f"Could not parse as NumPy: {str(e)}")
                else:
                    print_error(f"Failed to load {test_file}")
            else:
                print_warning("No .npy files found to test")
        else:
            print_warning("No files found to test load_single_image")
            
    except Exception as e:
        print_error(f"Error in load_single_image(): {str(e)}")

def test_get_image_stream():
    """Test get_image_stream function."""
    print_subheader("Testing get_image_stream()")
    try:
        # First get a file to test with
        files = load_dataset_folder('cifar100', 'clean')
        if files:
            # Find a .npy file
            npy_files = [f for f in files if f.endswith('.npy')]
            if npy_files:
                test_file = npy_files[0]
                stream = get_image_stream(test_file)
                
                if stream:
                    data = stream.read()
                    print_success(f"Successfully got stream for {test_file} ({len(data)} bytes)")
                    
                    # Try to load as numpy array
                    try:
                        numpy_array = np.load(io.BytesIO(data), allow_pickle=True)
                        print_info(f"Loaded NumPy array from stream with shape: {numpy_array.shape}")
                    except Exception as e:
                        print_warning(f"Could not parse stream as NumPy: {str(e)}")
                else:
                    print_error(f"Failed to get stream for {test_file}")
            else:
                print_warning("No .npy files found to test")
        else:
            print_warning("No files found to test get_image_stream")
            
    except Exception as e:
        print_error(f"Error in get_image_stream(): {str(e)}")

def test_load_imagenet_train():
    """Test load_imagenet_train function."""
    print_subheader("Testing load_imagenet_train()")
    try:
        classes = load_imagenet_train()
        if classes:
            print_success(f"Successfully loaded {len(classes)} ImageNet training classes")
            print_info(f"First 5 classes: {sorted(classes)[:5]}")
            print_info(f"Last 5 classes: {sorted(classes)[-5:]}")
        else:
            print_warning("No ImageNet training classes found")
            
    except Exception as e:
        print_error(f"Error in load_imagenet_train(): {str(e)}")

def test_load_cifar100_as_numpy():
    """Test load_cifar100_as_numpy function."""
    print_subheader("Testing load_cifar100_as_numpy()")
    try:
        # Test with clean folder
        images, labels = load_cifar100_as_numpy('clean')
        if images is not None and len(images) > 0:
            print_success(f"Successfully loaded CIFAR100 clean data as NumPy")
            print_info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            print_info(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
            
            # Visualize some samples
            visualize_numpy_data((images, labels), "CIFAR100 Clean Samples")
        else:
            print_warning("No CIFAR100 clean data found")
            
    except Exception as e:
        print_error(f"Error in load_cifar100_as_numpy(): {str(e)}")

def test_load_cifar100_meta():
    """Test load_cifar100_meta function."""
    print_subheader("Testing load_cifar100_meta()")
    try:
        meta = load_cifar100_meta()
        if meta:
            print_success("Successfully loaded CIFAR100 metadata")
            if isinstance(meta, dict):
                print_info(f"Metadata keys: {list(meta.keys())}")
                # Show some metadata content if available
                if 'fine_label_names' in meta:
                    print_info(f"Fine labels sample: {meta['fine_label_names'][:5]}")
                if 'coarse_label_names' in meta:
                    print_info(f"Coarse labels sample: {meta['coarse_label_names'][:5]}")
            else:
                print_info(f"Metadata type: {type(meta)}")
        else:
            print_warning("No CIFAR100 metadata found")
            
    except Exception as e:
        print_error(f"Error in load_cifar100_meta(): {str(e)}")

def test_load_dataset_split():
    """Test load_dataset_split function."""
    print_subheader("Testing load_dataset_split()")
    try:
        # Test CIFAR100 splits
        for split in ['train', 'test']:
            files = load_dataset_split('cifar100', split)
            if files:
                print_success(f"Successfully listed {len(files)} files in CIFAR100 {split} split")
                print_info(f"First 3 files: {files[:3]}")
            else:
                print_warning(f"No files found in CIFAR100 {split} split")
                
        # Test ImageNet splits if available
        for split in ['train', 'val']:
            try:
                files = load_dataset_split('imagenet', split)
                if files:
                    print_success(f"Successfully listed {len(files)} files in ImageNet {split} split")
                    print_info(f"First 3 files: {files[:3]}")
                else:
                    print_info(f"No files found in ImageNet {split} split")
            except:
                print_info(f"ImageNet {split} split not available")
                
    except Exception as e:
        print_error(f"Error in load_dataset_split(): {str(e)}")

def test_unpickle_from_s3():
    """Test unpickle_from_s3 function."""
    print_subheader("Testing unpickle_from_s3()")
    try:
        bucket = os.environ.get('S3_BUCKET_NAME')
        if not bucket:
            print_error("S3_BUCKET_NAME not set")
            return
            
        # Test with CIFAR100 train pickle
        try:
            train_data = unpickle_from_s3(bucket, 'cifar100/train')
            if train_data:
                print_success("Successfully unpickled CIFAR100 train data")
                if isinstance(train_data, dict):
                    print_info(f"Train data keys: {list(train_data.keys())}")
                    if b'data' in train_data:
                        print_info(f"Data shape: {train_data[b'data'].shape}")
                    if b'fine_labels' in train_data:
                        print_info(f"Number of labels: {len(train_data[b'fine_labels'])}")
                else:
                    print_info(f"Train data type: {type(train_data)}")
            else:
                print_warning("No data returned from unpickle")
        except Exception as e:
            print_warning(f"Could not unpickle CIFAR100 train: {str(e)}")
            
        # Test with CIFAR100 test pickle
        try:
            test_data = unpickle_from_s3(bucket, 'cifar100/test')
            if test_data:
                print_success("Successfully unpickled CIFAR100 test data")
                if isinstance(test_data, dict):
                    print_info(f"Test data keys: {list(test_data.keys())}")
            else:
                print_warning("No data returned from unpickle")
        except Exception as e:
            print_warning(f"Could not unpickle CIFAR100 test: {str(e)}")
            
        # Test with CIFAR100 meta
        try:
            meta_data = unpickle_from_s3(bucket, 'cifar100/meta')
            if meta_data:
                print_success("Successfully unpickled CIFAR100 meta data")
                if isinstance(meta_data, dict):
                    print_info(f"Meta data keys: {list(meta_data.keys())}")
            else:
                print_warning("No data returned from unpickle")
        except Exception as e:
            print_warning(f"Could not unpickle CIFAR100 meta: {str(e)}")
            
    except Exception as e:
        print_error(f"Error in unpickle_from_s3(): {str(e)}")

def run_all_tests():
    """Run all utility function tests."""
    print_header("TESTING S3 DATASET UTILITY FUNCTIONS")
    
    # Check environment first
    if not check_environment():
        return False
    
    # Run all tests
    test_load_dataset_numpy()
    test_load_cifar100_adversarial_or_clean()
    test_load_imagenet_adversarial_or_clean()
    test_get_dataset_config()
    test_load_dataset_folder()
    test_load_single_image()
    test_get_image_stream()
    test_load_imagenet_train()
    test_load_cifar100_as_numpy()
    test_load_cifar100_meta()
    test_load_dataset_split()
    test_unpickle_from_s3()
    
    print_header("ALL TESTS COMPLETED")
    return True

def main():
    """Main function with test menu."""
    print_header("S3 DATASET UTILS TEST SUITE")
    
    # Check environment
    if not check_environment():
        return
    
    # Test menu
    print_subheader("Test Menu")
    print("1.  Test load_dataset_numpy")
    print("2.  Test load_cifar100_adversarial_or_clean")
    print("3.  Test load_imagenet_adversarial_or_clean")
    print("4.  Test get_dataset_config")
    print("5.  Test load_dataset_folder")
    print("6.  Test load_single_image")
    print("7.  Test get_image_stream")
    print("8.  Test load_imagenet_train")
    print("9.  Test load_cifar100_as_numpy")
    print("10. Test load_cifar100_meta")
    print("11. Test load_dataset_split")
    print("12. Test unpickle_from_s3")
    print("13. Run all tests")
    print("14. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-14): ").strip()
        
        if choice == '1':
            test_load_dataset_numpy()
        elif choice == '2':
            test_load_cifar100_adversarial_or_clean()
        elif choice == '3':
            test_load_imagenet_adversarial_or_clean()
        elif choice == '4':
            test_get_dataset_config()
        elif choice == '5':
            test_load_dataset_folder()
        elif choice == '6':
            test_load_single_image()
        elif choice == '7':
            test_get_image_stream()
        elif choice == '8':
            test_load_imagenet_train()
        elif choice == '9':
            test_load_cifar100_as_numpy()
        elif choice == '10':
            test_load_cifar100_meta()
        elif choice == '11':
            test_load_dataset_split()
        elif choice == '12':
            test_unpickle_from_s3()
        elif choice == '13':
            run_all_tests()
        elif choice == '14':
            print("Exiting...")
            break
        else:
            print_error("Invalid choice! Please enter 1-14.")
        
        if choice in [str(i) for i in range(1, 14)]:
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