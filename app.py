from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from utilss.classes.cifar100 import Cifar100
from utilss.classes.imagenet import ImageNet
from utilss.classes.model import Model
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import os
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
cifer100_instance = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the BETTER API"}

@app.get("/check-dataset")
def check_cifar100():
    global cifer100_instance
    try:
        if cifer100_instance is None:
            cifer100_instance = Cifar100()
            cifer100_instance.load("cifar100")
            if not cifer100_instance.load("cifar100"):
                return {"message": "Failed to load CIFAR-100 dataset"}
        train_shape = cifer100_instance.x_train.shape
        test_shape = cifer100_instance.x_test.shape
        image_id = 19067  # Change this to test different image IDs
        image, label = cifer100_instance.get_train_image_by_id(image_id)
        original_model_filename = 'data/database/models/cifar100_resnet.keras'
        if os.path.exists(original_model_filename):
            original_model = tf.keras.models.load_model(original_model_filename)
            print(f'Model {original_model_filename} has been loaded')

        resnet_model = Model(original_model, 4, 0.8, original_model_filename)
        # resnet_model.model_accuracy(cifer100_instance.x_test, cifer100_instance.y_test)
        # print("Model accuracy: ", resnet_model.accuracy)
        selected_labels = ["pine_tree", "oak_tree", "willow_tree", "maple_tree", "forest","palm_tree", "man", "woman", "boy", "girl", "baby", "hamster", "shrew"]
        selected_accuracy, selected_loss = resnet_model.model_accuracy_selected(cifer100_instance, selected_labels)
        print("Model accuracy (selected labels): ", selected_accuracy)
        print("Model loss (selected labels): ", selected_loss)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
        return {
            "message": "CIFAR-100 dataset loaded successfully",
            "train_shape": train_shape,
            "test_shape": test_shape,
            "model_accuracy": selected_accuracy,
        }
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    
# def check_imagenet():
#     global imagenet_instance
#     try:
#         if imagenet_instance is None:
#             imagenet_instance = ImageNet()
#             if not imagenet_instance.load("imagenet"):
#                 return {"message": "Failed to load ImageNet dataset"}
        
#             weights = "imagenet"
#             model_filename = "resnet50_imagenet.keras"
#             top = 4
#             percent = 0.8
            
#         resnet_model = Model(ResNet50(weights=weights), top, percent, model_filename)
#         resnet_model.model_accuracy(imagenet_instance.test_data, imagenet_instance.labels)
#         print("Model accuracy: ", resnet_model.accuracy)
#         train_shape = (len(imagenet_instance.train_data),)  # Update to get the correct shape
#         test_shape = (len(imagenet_instance.test_data),)  # Update to get the correct shape
#         image_id = 66  # Change this to test different image IDs
#         image, label = imagenet_instance.get_train_image_by_id(image_id)
#         plt.imshow(image)
#         plt.title(f"Label: {label}")
#         plt.show()
#         return {
#             "message": "ImageNet dataset loaded successfully",
#             "train_shape": train_shape,
#             "test_shape": test_shape,
#             "model_accuracy": resnet_model.accuracy,
#         }
#     except Exception as e:
#         return {"message": f"An error occurred: {str(e)}"}
    
# @app.get("/get-image/{image_id}")
# def get_image(image_id: int):
#     global imagenet_instance
#     try:
#         if imagenet_instance is None:
#             return {"message": "ImageNet dataset is not loaded"}
        
#         image, label = imagenet_instance.get_train_image_by_id(image_id)
#         plt.imshow(image)
#         plt.title(f"Label: {label}")
#         plt.show()
#         return {
#             "message": "Image retrieved successfully",
#             "image_id": image_id,
#             "label": label,
#         }
#     except Exception as e:
#         return {"message": f"An error occurred: {str(e)}"}


handler = Mangum(app)