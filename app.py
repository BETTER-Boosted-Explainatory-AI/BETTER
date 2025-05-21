from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from dotenv import load_dotenv
import os

from utilss.exception_handlers import http_exception_handler, generic_exception_handler, validation_exception_handler

from routers.nma_router import nma_router
from routers.whitebox_testing_router import whitebox_testing_router
from routers.query_router import query_router
from routers.dataset_router import datasets_router
from routers.adversarial_router import adversarial_router
from routers.users_router import users_router
from routers.dendrogram_router import dendrogram_router
from routers.model_router import model_router

# import numpy as np

load_dotenv()

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Global variable to store the loaded ImageNet instance
imagenet_instance = None
cifer100_instance = None

FOLDER_PATH = os.getenv("PATH")

@app.get("/")
def read_root():
    return {"message": "Welcome to the BETTER API"}


app.include_router(nma_router)
app.include_router(model_router)
app.include_router(dendrogram_router)
app.include_router(whitebox_testing_router)
app.include_router(query_router)
app.include_router(datasets_router)
app.include_router(adversarial_router) 
app.include_router(users_router)


handler = Mangum(app)
