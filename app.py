from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from dotenv import load_dotenv
import os
from routers.hierarchical_clusters_router import hierarchical_clusters_router
from routers.confusion_matrix_router import confusion_matrix_router
from routers.whitebox_testing_router import whitebox_testing_router
from routers.sub_hierarchical_clusters_router import sub_hierarchical_clusters_router
from routers.query_router import query_router
from routers.dataset_router import datasets_router
# from routers.attacks.pgd_router import pgd_attack_router 
from routers.users_router import users_router
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
app.include_router(sub_hierarchical_clusters_router)
app.include_router(whitebox_testing_router)
app.include_router(query_router)
app.include_router(datasets_router)
# app.include_router(pgd_attack_router) 
app.include_router(users_router)


handler = Mangum(app)
