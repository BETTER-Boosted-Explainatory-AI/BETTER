import pytest
import boto3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
import json
from utilss.classes.user import User
import uuid
import os
import numpy as np
import io

from app import app
from services.users_service import require_authenticated_user
from .mock_data import user, models_metadata

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
def s3_models_json(mock_s3_bucket):
    user_id = user["user_id"]
    user_folder = f"{user_id}/"  # Adjust if your get_user_folder() returns something else
    models_json_key = f"{user_folder}models.json"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=models_json_key,
        Body=json.dumps(models_metadata['models']),
        ContentType="application/json"
    )

    # --- Add the model file ---
    model_file_key = f"{user_folder}35f658ac-aa29-461e-85fe-f7dcfe638dde/resnet50_imagenet.keras"
    mock_s3_bucket.put_object(
        Bucket="better-xai-users",
        Key=model_file_key,
        Body=b"dummy model content"
    )

    z_file_key = f"{user_folder}35f658ac-aa29-461e-85fe-f7dcfe638dde/similarity/dendrogram.pkl"
    mock_s3_bucket.put_object(
    Bucket="better-xai-users",
    Key=z_file_key,
    Body=b"dummy Z content"
    )

    return models_json_key

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

    for i in range(10):
        arr = np.array([i, i+1, i+2])
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        mock_s3_dataset_bucket.put_object(
            Bucket="better-datasets",
            Key=f"imagenet/clean/dummy{i+1}.npy",
            Body=buf.read(),
            ContentType="application/octet-stream"
        )

    # Add 10 adversarial .npy files
    for i in range(10):
        arr = np.array([i+10, i+11, i+12])
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        mock_s3_dataset_bucket.put_object(
            Bucket="better-datasets",
            Key=f"imagenet/adversarial/dummy{i+1}.npy",
            Body=buf.read(),
            ContentType="application/octet-stream"
        )

@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "better-xai-users"})
@patch("utilss.s3_utils.get_users_s3_client")
@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
@patch("tensorflow.keras.models.load_model")
def test_generate_adversarial_detector_success_with_images(
    mock_load_model,
    mock_get_user_models_info,
    mock_create_detector,
    mock_get_users_s3_client,
    client,
    mock_s3_bucket,
    s3_models_json,
    dataset_info_py
):
    mock_get_users_s3_client.return_value = mock_s3_bucket
    mock_create_detector.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", ("image1.npy", b"fakecontent1", "application/octet-stream")),
        ("clean_images", ("image2.npy", b"fakecontent2", "application/octet-stream")),
        ("adversarial_images", ("image3.npy", b"fakecontent3", "application/octet-stream")),
        ("adversarial_images", ("image4.npy", b"fakecontent4", "application/octet-stream")),
    ]

    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 201
    assert response.json()["result"] == "Detector created successfully"


@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "better-xai-users"})
@patch("utilss.s3_utils.get_users_s3_client")
@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
@patch("tensorflow.keras.models.load_model")
def test_generate_adversarial_detector_success_with_preprepared_images(
    mock_load_model,
    mock_get_user_models_info,
    mock_create_detector,
    mock_get_users_s3_client,
    client,
    mock_s3_bucket,
    s3_models_json,
    dataset_info_py
):
    mock_get_users_s3_client.return_value = mock_s3_bucket
    mock_create_detector.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }

    response = client.post("/api/adversarial/generate", data=data)
    assert response.status_code == 201
    assert response.json()["result"] == "Detector created successfully"

@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "better-xai-users"})
@patch("utilss.s3_utils.get_users_s3_client")
@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
@patch("tensorflow.keras.models.load_model")
def test_generate_adversarial_detector_error_not_enough_images_err(
    mock_load_model,
    mock_get_user_models_info,
    mock_create_detector,
    mock_get_users_s3_client,
    client,
    mock_s3_bucket,
    s3_models_json,
    dataset_info_py
):
    mock_get_users_s3_client.return_value = mock_s3_bucket
    mock_create_detector.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", ("image1.npy", b"fakecontent1", "application/octet-stream")),
        ("adversarial_images", ("image3.npy", b"fakecontent3", "application/octet-stream")),
    ]

    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 500
    assert "This solver needs samples of at least 2 classes in the data" in response.json()["detail"]

@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "better-xai-users"})
@patch("utilss.s3_utils.get_users_s3_client")
@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.models_service.get_user_models_info", return_value=models_metadata['models'])
@patch("tensorflow.keras.models.load_model")
def test_generate_adversarial_detector_not_npy_images_err(
    mock_load_model,
    mock_get_user_models_info,
    mock_create_detector,
    mock_get_users_s3_client,
    client,
    mock_s3_bucket,
    s3_models_json,
    dataset_info_py
):
    mock_get_users_s3_client.return_value = mock_s3_bucket
    mock_create_detector.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", ("image1.jpg", b"fakecontent1", "application/octet-stream")),
        ("clean_images", ("image2.jpg", b"fakecontent2", "application/octet-stream")),
        ("adversarial_images", ("image3.jpg", b"fakecontent3", "application/octet-stream")),
        ("adversarial_images", ("image4.jpg", b"fakecontent4", "application/octet-stream")),
    ]

    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 422
    assert "Only .npy files are allowed." in response.json()["detail"]