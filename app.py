from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from utilss.classes.cifar100 import Cifar100
from utilss.classes.imagenet import ImageNet
import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded ImageNet instance
imagenet_instance = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the BETTER API"}

@app.get("/check-dataset")
# def check_cifar100():
#     cifar100 = Cifar100()
#     cifar100.load("cifar100")
#     if cifar100.x_train is not None and cifar100.y_train is not None:
#         image_id = 43  # Change this to test different image IDs
#         try:
#             image, label = cifar100.get_train_image_by_id(image_id)
#             plt.imshow(image)
#             plt.title(f"Label: {label}")
#             plt.show()
#             return {
#                 "message": "CIFAR-100 dataset loaded successfully",
#                 "image_id": int(image_id),  # Convert to native Python int
#                 "image_shape": list(image.shape),  # Convert NumPy shape to list
#                 "label": label,  
#             }
#         except ValueError as e:
#             return {"error": str(e)}
#     else:
#         return {"message": "Failed to load CIFAR-100 dataset"}
    
def check_imagenet():
    global imagenet_instance
    try:
        if imagenet_instance is None:
            imagenet_instance = ImageNet()
            if not imagenet_instance.load("imagenet"):
                return {"message": "Failed to load ImageNet dataset"}
        
        train_shape = (len(imagenet_instance.train_data),)  # Update to get the correct shape
        test_shape = (len(imagenet_instance.test_data),)  # Update to get the correct shape
        image_id = 66  # Change this to test different image IDs
        image, label = imagenet_instance.get_train_image_by_id(image_id)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
        return {
            "message": "ImageNet dataset loaded successfully",
            "train_shape": train_shape,
            "test_shape": test_shape,
        }
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    
@app.get("/get-image/{image_id}")
def get_image(image_id: int):
    global imagenet_instance
    try:
        if imagenet_instance is None:
            return {"message": "ImageNet dataset is not loaded"}
        
        image, label = imagenet_instance.get_train_image_by_id(image_id)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
        return {
            "message": "Image retrieved successfully",
            "image_id": image_id,
            "label": label,
        }
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


handler = Mangum(app)