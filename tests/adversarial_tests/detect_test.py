import pytest
import boto3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pickle
import moto
import json
from utilss.classes.user import User
import uuid
import os
import numpy as np
import io
from PIL import Image

from app import app
from utilss.classes.score_calculator import ScoreCalculator
from services.users_service import require_authenticated_user
from .mock_data import user, models_metadata, labels

class DummyDetector:
    def predict(self, X):
        return [0] * len(X)
    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]

detector = DummyDetector()
threshold = 0.5  # or whatever threshold you want

detector_bytes = pickle.dumps((detector, threshold)) 

Z = [
    [0, 1, 0.5, 2],
    [2, 3, 1.0, 3]
]

buffer = io.BytesIO()
pickle.dump(Z, buffer)
buffer.seek(0)

# reusable client fixture for testing
@pytest.fixture
def client():
    return TestClient(app)

# pytest fixture to override the authentication dependency
@pytest.fixture(scope="function", autouse=True)
def override_auth_dep():
    def mock_user():
        return User(user_id=uuid.UUID(user["user_id"]), email=user["email"])
    app.dependency_overrides[require_authenticated_user] = mock_user
    yield
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def mock_aws():
    mock = moto.mock_aws()
    mock.start()
    yield mock
    mock.stop()

@pytest.fixture(scope="function")
def mock_s3_bucket(mock_aws):
    s3 = boto3.client("s3", region_name="eu-west-1")
    bucket_name = "better-xai-users"
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    return s3

@pytest.fixture
def mock_user_files(mock_s3_bucket):
    user_id = user["user_id"]
    user_folder = f"{user_id}/"  # Adjust if your get_user_folder() returns something else
    models_json_key = f"{user_folder}models.json"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=models_json_key,
        Body=json.dumps(models_metadata['models']),
        ContentType="application/json"
    )

    model_file_key = f"{user_folder}35f658ac-aa29-461e-85fe-f7dcfe638dde/resnet50_imagenet.keras"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=model_file_key,
        Body=b"dummy model content"
    )

    detector_file_key = f"{user_folder}35f658ac-aa29-461e-85fe-f7dcfe638dde/similarity/test_detector.pkl"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=detector_file_key,
        Body=detector_bytes,
    )

    Z_file_key = f"{user_folder}35f658ac-aa29-461e-85fe-f7dcfe638dde/similarity/dendrogram.pkl"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=Z_file_key,
        Body=buffer.read()
    )


@pytest.fixture(scope="function")
def mock_s3_dataset_bucket(mock_aws):
    s3 = boto3.client("s3", region_name="eu-west-1")
    bucket_name = "better-datasets"
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    return s3

@pytest.fixture
def dataset_info_py(mock_s3_dataset_bucket):
    # Get the path to imagenet_info.py (adjust the path if needed)
    current_dir = os.path.dirname(__file__)
    imagenet_info_path = os.path.join(current_dir, "imagenet_info.py")
    with open(imagenet_info_path, "r", encoding="utf-8") as f:
        py_content = f.read()

    mock_s3_dataset_bucket.put_object(
        Bucket="better-datasets",
        Key="imagenet_info.py",
        Body=py_content,
        ContentType="text/x-python"
    )

@patch.object(ScoreCalculator, "calculate_adversarial_score", return_value=42)
@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "better-xai-users"})
@patch("utilss.s3_utils.get_users_s3_client")
@patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
@patch("tensorflow.keras.models.load_model")
@patch("services.dataset_service.get_dataset_labels", return_value=labels)
def test_image_detection_success(
    mock_get_dataset_labels,
    mock_load_model,
    mock_get_user_models_info,
    mock_get_users_s3_client,
    mock_calc_score,  
    client,
    mock_s3_bucket,
    mock_user_files,
    mock_s3_dataset_bucket,
    dataset_info_py,
):
    mock_get_users_s3_client.return_value = mock_s3_bucket
    mock_load_model.return_value = MagicMock()

    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity",
        "detector_filename": "test_detector.pkl"
    }

    img = Image.new("RGB", (1, 1), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    image_content = img_bytes.read()

    files = [("image", ("image1.png", image_content, "image/png"))]

    response = client.post("/api/adversarial/detect", data=data, files=files)
    assert response.status_code == 200
    assert "result" in response.json()

# @patch("utilss.s3_utils.get_users_s3_client")
# @patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
# @patch("services.adversarial_attacks_service.detect_adversarial_image")
# def test_image_detection_success(
#     mock_detect_adversarial_image,
#     mock_get_user_models_info,
#     mock_get_users_s3_client,
#     client,  # <-- pytest fixture now in the correct position
#     mock_s3_bucket,
#     mock_user_files,
# ):
#     mock_detect_adversarial_image.return_value = {
#         "image": "fake_base64",
#         "predictions": [{"label": "cat", "prob": 0.9}],
#         "result": "Clean",
#         "probability": 0.1
#     }

#     data = {
#         "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
#         "graph_type": "similarity",
#         "detector_filename": "test_detector.pkl"
#     }

#     img = Image.new("RGB", (1, 1), color="white")
#     img_bytes = io.BytesIO()
#     img.save(img_bytes, format="PNG")
#     img_bytes.seek(0)
#     image_content = img_bytes.read()

#     files = [("image", ("image1.png", image_content, "image/png"))]

#     # ...rest of your test...
#     response = client.post("/api/adversarial/detect", data=data, files=files)
#     assert response.status_code == 200
#     assert response.json()["result"] == "Clean"