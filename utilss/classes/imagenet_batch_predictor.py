import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ImageNetBatchPredictor:
    def __init__(self, model, batch_size=64, img_size=(224, 224), num_workers=16):
        self.model = model
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.img_cache = {}  # Simple memory cache for processed images
        
    def _preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        # Check cache first
        if img_path in self.img_cache:
            return self.img_cache[img_path]
            
        try:
            img = load_img(img_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            
            # Store in cache
            self.img_cache[img_path] = img_array
            
            return img_array
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None
    
    def _preprocess_batch_parallel(self, batch_paths):
        """Preprocess a batch of images in parallel"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Process images in parallel
            results = list(executor.map(self._preprocess_image, batch_paths))
        
        # Filter out None values and keep track of valid paths
        valid_results = []
        valid_paths = []
        
        for i, result in enumerate(results):
            if result is not None:
                valid_results.append(result)
                valid_paths.append(batch_paths[i])
                
        return valid_paths, valid_results
        
    def get_top_predictions(self, image_paths, top_k=5):
        """
        Process batches of image paths and return the top predictions for each image.
        Uses parallel processing for image preprocessing.
        """
        results = []
        
        # Process images in super-batches to amortize overhead
        super_batch_size = self.batch_size * 4  # Larger batches for parallel processing
        
        for i in range(0, len(image_paths), super_batch_size):
            super_batch_paths = image_paths[i:i+super_batch_size]
            
            # Preprocess the super-batch in parallel
            valid_paths, valid_images = self._preprocess_batch_parallel(super_batch_paths)
            
            if not valid_images:
                continue
                
            # Split into actual prediction batches
            for j in range(0, len(valid_images), self.batch_size):
                batch_end = min(j + self.batch_size, len(valid_images))
                batch_images = valid_images[j:batch_end]
                batch_paths = valid_paths[j:batch_end]
                
                # Convert to batch tensor
                batch_tensor = np.array(batch_images)
                
                # Get predictions for the entire batch at once
                with tf.device('/GPU:0'):  # Explicitly use GPU
                    batch_preds = self.model.predict(batch_tensor, verbose=0)
                
                # Process predictions for each image in the batch
                for k, preds in enumerate(batch_preds):
                    # Get decoded predictions (class names and probabilities)
                    decoded_preds = decode_predictions(np.expand_dims(preds, axis=0), top=top_k)[0]
                    
                    # Add original path to results
                    results.append((batch_paths[k], decoded_preds))
        
        return results
