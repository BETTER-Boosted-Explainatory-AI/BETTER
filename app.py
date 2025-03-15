from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from dotenv import load_dotenv
import os

from data.datasets.cifar100_info import CIFAR100_INFO
from data.datasets.imagenet_info import IMAGENET_INFO

from utilss.classes.model import Model
# from utilss.classes.datasets.cifar100 import Cifar100
# from utilss.classes.datasets.imagenet import ImageNet
from utilss.classes.datasets.dataset_factory import DatasetFactory
from utilss.classes.preprocessing.prediction_graph import PredictionGraph
from utilss.classes.preprocessing.edges_dataframe import EdgesDataframe
from utilss.classes.heap_processor import HeapGraphProcessor

import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

from routers.hierarchical_clusters_router import hierarchical_clusters_router

# from PIL import Image
import numpy as np

load_dotenv()

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

FOLDER_PATH = os.getenv("PATH")



@app.get("/")
def read_root():
    return {"message": "Welcome to the BETTER API"}


app.include_router(hierarchical_clusters_router)

# @app.get("/graph")
# def create_graph():

#     imagenet_dataset = DatasetFactory.create_dataset(IMAGENET_INFO["dataset"])
#     imagenet_dataset.load(IMAGENET_INFO["dataset"])
    
#     try:
#         graph_types = ["similarity", "dissimilarity", "confusion_matrix"]
#         heap_type = ["min", "max"]
        
#         weights = "imagenet"
#         model_filename = f'data/graphs/resnet50_mini_imagenet.keras'
#         dataframe_filename = f'data/graphs/edges_{graph_types[1]}_mini_imagenet.csv'
#         graph_filename = f'data/graphs/graph_{graph_types[1]}_mini_imagenet.graphml'
#         trainset_path = 'data/datasets/imagenet/train'

#         resnet_model_imagenet = Model(
#             ResNet50(weights=weights), IMAGENET_INFO["top_k"], IMAGENET_INFO["min_confidence"], model_filename, IMAGENET_INFO["dataset"]
#         )

#         count_grpah = PredictionGraph(model_filename, graph_filename, graph_types[1], IMAGENET_INFO["labels"], IMAGENET_INFO["threshold"], IMAGENET_INFO["infinity"])
#         edges_df_temp = count_grpah.create_graph_imagenet(resnet_model_imagenet,IMAGENET_INFO["top_k"], trainset_path, IMAGENET_INFO["labels_dict"])
#         edges_df = EdgesDataframe(resnet_model_imagenet, dataframe_filename, edges_df_temp)
#         count_grpah.save_graph()
#         edges_df.save_dataframe()
#     except Exception as e:
#         return {"message": f"An error occurred: {str(e)}"}


@app.get("/check-dataset")
def check_cifar100():
    
    # global cifer100_instance
    try:
        #     if cifer100_instance is None:
        #         cifer100_instance = Cifar100()
        #         cifer100_instance.load("cifar100")
        #         if not cifer100_instance.load("cifar100"):
        #             return {"message": "Failed to load CIFAR-100 dataset"}
        original_model_filename = "data/database/models/cifar100_resnet.keras"
        if os.path.exists(original_model_filename):
            original_model = tf.keras.models.load_model(original_model_filename)
            print(f"Model {original_model_filename} has been loaded")
            resnet_model = Model(
                original_model, 4, 0.8, original_model_filename, "cifar100"
            )
            image_path = "data/database/photos/cat.jpg"
            images_paths = [
                "data/database/photos/cat.jpg",
                "data/database/photos/pug.jfif",
            ]
            prediction = resnet_model.predict_batches(images_paths)
            print(f"Prediction: {prediction}")
            # resnet_model.model_f1score(cifer100_instance.x_test, cifer100_instance.y_test)
            # selected_labels = ["pine_tree", "oak_tree", "willow_tree", "maple_tree", "forest","palm_tree", "man", "woman", "boy", "girl", "baby", "hamster", "shrew"]
            # selected_f1 = resnet_model.model_f1score_selected(cifer100_instance, selected_labels)
        return {
            "message": "CIFAR-100 dataset loaded successfully",
            # "f1score": resnet_model.f1score,
            # "selected_f1": selected_f1,
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

#         weights = "imagenet"
#         model_filename = "resnet50_imagenet.keras"
#         top = 4
#         percent = 0.8

# resnet_model_imagenet = Model(ResNet50(weights=weights), top, percent, model_filename, "imagenet")
# resnet_model_imagenet.model_evaluate_imagenet("data/datasets/imagenet/test")
# print("Model accuracy: ", resnet_model_imagenet.accuracy)
# train_shape = (len(imagenet_instance.train_data),)  # Update to get the correct shape
# test_shape = (len(imagenet_instance.test_data),)  # Update to get the correct shape
# image_id = 65  # Change this to test different image IDs
# image, label = imagenet_instance.get_train_image_by_id(image_id)
# image_path = "data/database/photos/cat.jpg"
# images_paths = ["data/database/photos/cat.jpg", "data/database/photos/pug.jfif"]
# prediction = resnet_model_imagenet.predict_batches(images_paths)
# print(f"Prediction: {prediction}")
# plt.imshow(image)
# plt.title(f"Label: {label}")
# plt.show()
# return {
#     "message": "ImageNet dataset loaded successfully",
# "prediction": prediction,
# "train_shape": train_shape,
# "test_shape": test_shape,
# "model_accuracy": resnet_model_imagenet.accuracy,
#     }
# except Exception as e:
#     return {"message": f"An error occurred: {str(e)}"}

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
