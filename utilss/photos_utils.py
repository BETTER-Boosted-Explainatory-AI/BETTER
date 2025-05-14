import numpy as np
import tensorflow as tf
from services.models_service import get_cached_preprocess_function
from PIL import Image
import io
import base64


def preprocess_loaded_image(model, image):
    expected_shape = model.input_shape
    input_height, input_width = expected_shape[1], expected_shape[2]
    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    pil_image = pil_image.resize((input_width, input_height))
    preprocess_input = get_cached_preprocess_function(model)
    image_array = preprocess_input(np.array(pil_image))
    image_preprocessed = np.expand_dims(image_array, axis=0)
    return image_preprocessed

def preprocess_image(model, image):
    preprocess_input = get_cached_preprocess_function(model)
    image_array = preprocess_input(np.array(image))
    image_preprocessed = np.expand_dims(image_array, axis=0)
    return image_preprocessed

def preprocess_deepfool_image(model, image):
    expected_shape = model.input_shape
    input_height, input_width = expected_shape[1], expected_shape[2]
    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    pil_image = pil_image.resize((input_width, input_height))
    img = tf.keras.preprocessing.image.img_to_array(pil_image)
    norm_image = np.array(img)
    norm_image = norm_image / 255.0
    return np.expand_dims(norm_image, axis=0)

def deprocess_resnet_image(processed_img):
    img = processed_img
    if len(img.shape) == 4:
        img = img[0]  # Remove batch dimension
        
    # Reverse the ResNet preprocessing
    # Add back the means
    img = img + np.array([103.939, 116.779, 123.68])
    
    # BGR to RGB
    img = img[:, :, ::-1]
    
    # Clip values to valid range and convert to uint8
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def preprocess_numpy_image(model, image):
    """
    Preprocess a NumPy array image for the given model.
    """
    if image.ndim == 3:
        # If the image is 3D, add a batch dimension
        image = np.expand_dims(image, axis=0)

    # Get the appropriate preprocessing function for the model
    preprocess_input = get_cached_preprocess_function(model)

    # Apply the preprocessing function
    image_preprocessed = preprocess_input(image)

    return image_preprocessed

def encode_image_to_base64(image):
    """
    Encode a NumPy array image to a Base64 string.
    """
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8)) if image.max() <= 1 else Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, tf.Tensor):
        # Convert Tensor to NumPy array
        image = Image.fromarray(image.numpy().astype(np.uint8))

    # Save the image to a BytesIO buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode the image to Base64
    print(f"Encoding image to base64")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64
