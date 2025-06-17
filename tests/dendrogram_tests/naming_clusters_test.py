import pytest
import boto3
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
from utilss.classes.user import User
import uuid
import json

from app import app
from services.users_service import require_authenticated_user
from .mock_data import user, mock_dendrogram, selected_labels, mock_dendrogram_after_name_change, mock_sub_dendrogram_after_name_change

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
        "selected_labels": selected_labels,
        "cluster_id": 1000,
        "new_name": "tubs_1"
    }
## status code 200 - OK
def test_renaming_cluster(client, mock_s3_bucket, dendrogram_request):
    """Test successful cluster renaming in dendrograms."""
    # Mock the model path check
    with patch("services.dendrogram_service._check_model_path") as mock_check_path:
        mock_check_path.return_value = "test/path"
        
        # Mock the Dendrogram class
        mock_dendrogram_obj = MagicMock()
        mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj  # Return self
        mock_dendrogram_obj.Z_tree_format = mock_dendrogram  # Set the tree format
        
        # Mock rename_cluster to return the mock subtree after name change
        mock_dendrogram_obj.rename_cluster.return_value = mock_dendrogram_after_name_change
        
        # Mock get_sub_dendrogram_formatted to return a JSON string
        mock_dendrogram_obj.get_sub_dendrogram_formatted.return_value = json.dumps(mock_sub_dendrogram_after_name_change)
        
        # Mock the Dendrogram class creation
        with patch("services.dendrogram_service.Dendrogram", return_value=mock_dendrogram_obj):
            response = client.put("/api/dendrograms/naming_clusters", json=dendrogram_request)

            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "id" in data
            assert "name" in data
            assert "children" in data
            
            # Verify the mocked data was returned
            assert data["id"] == mock_sub_dendrogram_after_name_change["id"]
            assert data["name"] == mock_sub_dendrogram_after_name_change["name"]
            assert len(data["children"]) == len(mock_sub_dendrogram_after_name_change["children"])
            
            # Verify the service chain was called correctly
            mock_check_path.assert_called_once_with(
                uuid.UUID(user["user_id"]),
                uuid.UUID(dendrogram_request["model_id"]),
                dendrogram_request["graph_type"]
            )
            mock_dendrogram_obj.load_dendrogram.assert_called_once()
            mock_dendrogram_obj.rename_cluster.assert_called_once_with(
                dendrogram_request["cluster_id"],
                dendrogram_request["new_name"]
            )
            mock_dendrogram_obj.get_sub_dendrogram_formatted.assert_called_once_with(
                dendrogram_request["selected_labels"]
            )

## status code 404 - not found
def test_renaming_cluster_not_found(client, mock_s3_bucket, dendrogram_request):
    """Test handling of non-existent dendrogram"""
    with patch("services.dendrogram_service._check_model_path") as mock_check_path:
        mock_check_path.return_value = None

        response = client.put("/api/dendrograms/naming_clusters", json=dendrogram_request)

        assert response.status_code == 404
        assert response.json()["detail"] == "Model path not found"

## status code 422 - unprocessable entity
def test_renaming_cluster_invalid_request(client, mock_s3_bucket):
    """Test handling of invalid request data"""
    # Missing required field
    invalid_request = {
        "model_id": "1be86594-1de3-4dcd-bc25-560f5653690f",
        "graph_type": "similarity",
        "selected_labels": selected_labels,
        "cluster_id": 1000
        # Missing new_name
    }

    response = client.put("/api/dendrograms/naming_clusters", json=invalid_request)
    assert response.status_code == 422  # Validation error

## status code 500 - internal server error
def test_renaming_cluster_internal_error(client, mock_s3_bucket, dendrogram_request):
    """Test handling of internal server errors during cluster renaming"""
    with patch("services.dendrogram_service._check_model_path") as mock_check_path:
        mock_check_path.return_value = "test/path"
        
        # Mock the Dendrogram class to raise an exception
        mock_dendrogram_obj = MagicMock()
        mock_dendrogram_obj.load_dendrogram.side_effect = Exception("Unexpected error")
        mock_dendrogram_obj.Z_tree_format = mock_dendrogram  # Set the tree format
        
        with patch("services.dendrogram_service.Dendrogram", return_value=mock_dendrogram_obj):
            response = client.put("/api/dendrograms/naming_clusters", json=dendrogram_request)
            
            assert response.status_code == 500
            assert "detail" in response.json()
            assert "Unexpected error" in response.json()["detail"]
