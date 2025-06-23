import pytest
import boto3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
import json
from utilss.classes.user import User
import uuid

from app import app
from services.users_service import require_authenticated_user
from .mock_data import user

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
    bucket_name = "test-bucket"
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    return s3

@pytest.fixture
def fake_upload_file():
    class FakeUploadFile:
        def __init__(self, filename):
            self.filename = filename
    return FakeUploadFile

@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"})
def test_generate_adversarial_detector_success(
    mock_create_detector, client, mock_s3_bucket, fake_upload_file
):
    mock_create_detector.return_value = MagicMock()
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", (fake_upload_file("image1.npy").filename, b"fakecontent", "application/octet-stream")),
        ("adversarial_images", (fake_upload_file("image2.npy").filename, b"fakecontent", "application/octet-stream"))
    ]
    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 201
    assert response.json()["result"] == "Detector created successfully"

@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"})
def test_generate_adversarial_detector_invalid_file_type(
    mock_create_detector, client, mock_s3_bucket, fake_upload_file
):
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", (fake_upload_file("image1.jpg").filename, b"fakecontent", "application/octet-stream"))
    ]
    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 422
    assert "Invalid file type for clean_images" in response.json()["detail"]


@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"})
def test_generate_adversarial_detector_detector_not_created(
    mock_create_detector, client, mock_s3_bucket
):
    mock_create_detector.return_value = None
    data = {
        "current_model_id": "35f658ac-aa29-461e-85fe-f7dcfe638dde",
        "graph_type": "similarity"
    }
    response = client.post("/api/adversarial/generate", data=data)
    assert response.status_code == 404
    assert response.json()["detail"] == "Detector was not created"