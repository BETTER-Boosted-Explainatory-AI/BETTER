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
from utilss.classes.preprocessing.heap_processor import HeapGraphProcessor

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

handler = Mangum(app)
