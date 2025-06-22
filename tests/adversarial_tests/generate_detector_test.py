import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app import app

client = TestClient(app)

class MockUser:
    user_id = "02f58494-a0c1-7003-7f34-2e1be083c8fa"
    email = "ixMomozz@gmail.com"

@pytest.fixture
def fake_user():
    class FakeUser:
        user_id = "test-user-id"
        def get_user_folder(self):
            return "test-user-folder"
    return FakeUser()

@pytest.fixture
def fake_upload_file():
    class FakeUploadFile:
        def __init__(self, filename):
            self.filename = filename
    return FakeUploadFile

@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.users_service.require_authenticated_user") 
def test_generate_adversarial_detector_success(mock_create_detector, mock_require_user, fake_user, fake_upload_file):
    mock_require_user.return_value = fake_user
    mock_create_detector.return_value = MagicMock()
    data = {
        "current_model_id": "model123",
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
@patch("services.users_service.require_authenticated_user") 
def test_generate_adversarial_detector_invalid_file_type(mock_create_detector, mock_require_user, fake_user, fake_upload_file):
    mock_require_user.return_value = fake_user
    data = {
        "current_model_id": "model123",
        "graph_type": "similarity"
    }
    files = [
        ("clean_images", (fake_upload_file("image1.jpg").filename, b"fakecontent", "application/octet-stream"))
    ]
    response = client.post("/api/adversarial/generate", data=data, files=files)
    assert response.status_code == 422
    assert "Invalid file type for clean_images" in response.json()["detail"]


@patch("services.adversarial_attacks_service.create_logistic_regression_detector")
@patch("services.users_service.require_authenticated_user") 
def test_generate_adversarial_detector_detector_not_created(mock_create_detector, mock_require_user, fake_user):
    mock_require_user.return_value = fake_user
    mock_create_detector.return_value = None
    data = {
        "current_model_id": "model123",
        "graph_type": "similarity"
    }
    response = client.post("/api/adversarial/generate", data=data)
    assert response.status_code == 404
    assert response.json()["detail"] == "Detector was not created"