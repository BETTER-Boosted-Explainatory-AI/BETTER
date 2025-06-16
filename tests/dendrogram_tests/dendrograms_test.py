import pytest
import boto3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
import json
from utilss.classes.user import User
import uuid

from app import app
from routers.dendrogram_router import dendrogram_router
from services.users_service import require_authenticated_user
from request_models.dendrogram_model import DendrogramRequest, DendrogramResult
from .mock_data import user, mock_dendrogram, mock_sub_dendrogram, selected_labels

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

# Fixture for common request body
@pytest.fixture
def dendrogram_request():
    return {
        "model_id": "1be86594-1de3-4dcd-bc25-560f5653690f",
        "graph_type": "similarity",
        "selected_labels": selected_labels
    }

## status code 200 - OK
def test_get_dendrogram_success(client, mock_s3_bucket, dendrogram_request):
    """Test successful retrieval of a sub-dendrogram"""
    # Mock the model path check
    with patch("services.dendrogram_service._check_model_path") as mock_check_path:
        mock_check_path.return_value = "test/path"
        
        # Mock the Dendrogram class
        mock_dendrogram_obj = MagicMock()
        mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj  # Return self
        mock_dendrogram_obj.Z_tree_format = mock_dendrogram  # Set the tree format
        
        # Mock get_sub_dendrogram_formatted to return a JSON string
        def mock_get_sub_dendrogram_formatted(labels):
            return json.dumps(mock_sub_dendrogram)
        mock_dendrogram_obj.get_sub_dendrogram_formatted.side_effect = mock_get_sub_dendrogram_formatted
        
        with patch("services.dendrogram_service.Dendrogram", return_value=mock_dendrogram_obj):
            response = client.post("/api/dendrograms", json=dendrogram_request)

            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "id" in data
            assert "name" in data
            assert "children" in data
            assert "selected_labels" in data
            
            # Verify the mocked data was returned
            assert data["id"] == mock_sub_dendrogram["id"]
            assert data["name"] == mock_sub_dendrogram["name"]
            assert data["selected_labels"] == selected_labels
            assert len(data["children"]) == len(mock_sub_dendrogram["children"])
            
            # Verify the service chain was called correctly
            mock_check_path.assert_called_once_with(
                uuid.UUID(user["user_id"]),
                uuid.UUID(dendrogram_request["model_id"]),
                dendrogram_request["graph_type"]
            )
            mock_dendrogram_obj.load_dendrogram.assert_called_once()
            mock_dendrogram_obj.get_sub_dendrogram_formatted.assert_called_once_with(
                dendrogram_request["selected_labels"]
            )

## status code 404 - not found
def test_get_dendrogram_not_found(client, mock_s3_bucket, dendrogram_request):
    """Test handling of non-existent dendrogram"""
    with patch("services.dendrogram_service._check_model_path") as mock_check_path:
        mock_check_path.return_value = None

        response = client.post("/api/dendrograms", json=dendrogram_request)

        assert response.status_code == 404
        assert response.json()["detail"] == "Model path not found"

## status code 422 - unprocessable entity
def test_get_dendrogram_invalid_request(client, mock_s3_bucket):
    """Test handling of invalid request data"""
    # Missing required field
    invalid_request = {
        "graph_type": "similarity",
        "selected_labels": selected_labels
    }

    response = client.post("/api/dendrograms", json=invalid_request)
    assert response.status_code == 422  # Validation error

