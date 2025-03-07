from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from utils.classes.cifar100 import Cifar100
# from utils.classes.imagenet import ImageNet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the BETTER API"}

@app.get("/check-dataset")
def check_cifar100():
    cifar100 = Cifar100()
    cifar100.load("cifar100")
    if cifar100.x_train is not None and cifar100.y_train is not None:
        image_id = 43  # Change this to test different image IDs
        try:
            image, label = cifar100.get_train_image_by_id(image_id)
            plt.imshow(image)
            plt.title(f"Label: {label}")
            plt.show()
            return {
                "message": "CIFAR-100 dataset loaded successfully",
                "image_id": int(image_id),  # Convert to native Python int
                "image_shape": list(image.shape),  # Convert NumPy shape to list
                "label": label,  
            }
        except ValueError as e:
            return {"error": str(e)}
    else:
        return {"message": "Failed to load CIFAR-100 dataset"}
    
# def check_imagenet():
#     try:
#         imagenet = ImageNet()
#         if imagenet.load("imagenet"):
#             train_shape = imagenet.train_data.shape
#             test_shape = imagenet.test_data.shape
#             return {
#                 "message": "ImageNet dataset loaded successfully",
#                 "train_shape": train_shape,
#                 "test_shape": test_shape,
#             }
#         else:
#             return {"message": "Failed to load ImageNet dataset"}
#     except Exception as e:
#         return {"message": f"An error occurred: {str(e)}"}


handler = Mangum(app)