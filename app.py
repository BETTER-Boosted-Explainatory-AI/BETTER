from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from dotenv import load_dotenv
import os
from routers.hierarchical_clusters_router import hierarchical_clusters_router
from routers.confusion_matrix_router import confusion_matrix_router
from routers.whitebox_testing_router import whitebox_testing_router

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
app.include_router(confusion_matrix_router)
app.include_router(whitebox_testing_router)

handler = Mangum(app)
