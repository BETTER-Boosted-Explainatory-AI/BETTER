import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
from utilss.classes.user import User
import uuid
from pathlib import Path

from app import app
from services.users_service import require_authenticated_user
from .query_mock_data import user, models_metadata, model_id, verbal_explanation, top_label, top_predictions, labels, mock_dendrogram

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

@pytest.fixture
def test_image_path():
    # Get the directory where this test file is located
    current_dir = Path(__file__).parent
    image_path = current_dir / "test_image.png"
    
    # Create a simple test image if it doesn't exist
    if not image_path.exists():
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
    
    return str(image_path)

@pytest.fixture
def query_request(test_image_path):
    return {
        "current_model_id": model_id,
        "graph_type": models_metadata['models'][0]['graph_type'][0],
        "image": test_image_path
    }

def test_query_success(client, query_request, test_image_path, mock_aws):
    """Test successful query with a real image file"""
    # Mock the model service functions
    with patch("services.models_service.get_user_models_info") as mock_get_model_info:
        mock_get_model_info.return_value = models_metadata['models'][0]
        
        # Mock the model service functions
        with patch("services.models_service.load_model_from_s3") as mock_load_model, \
             patch("services.models_service.construct_model") as mock_construct_model:
            mock_load_model.return_value = MagicMock()
            mock_construct_model.return_value = MagicMock()
            
            # Mock the S3 file existence check
            with patch("services.models_service.s3_file_exists") as mock_s3_file_exists:
                mock_s3_file_exists.return_value = True
            
            with patch("services.models_service.get_model_specific_file") as mock_get_model_specific_file:
                mock_get_model_specific_file.return_value = "test/path"
            
            with patch("services.dataset_service.get_dataset_labels") as mock_get_dataset_labels:
                mock_get_dataset_labels.return_value = labels
            
            with patch("services.models_service.get_user_models_info") as mock_get_user_models_info:
                mock_get_user_models_info.return_value = models_metadata
            
            with patch("services.models_service.query_model") as mock_query_model:
                mock_query_model.return_value = "test/path"
            
            with patch("services.models_service.query_predictions") as mock_query_predictions:
                mock_query_predictions.return_value = top_label, verbal_explanation
            
            with patch("services.models_service.get_model_specific_file") as mock_get_model_specific_file:
                mock_get_model_specific_file.return_value = "test/path"
            
            with patch("services.models_service.get_top_k_predictions") as mock_get_top_k_predictions:
                mock_get_top_k_predictions.return_value = top_predictions
            
            with patch("utilss.classes.dendrogram.Dendrogram") as mock_dendrogram_class:
                mock_dendrogram_obj = MagicMock()
                mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj  # Return self
                mock_dendrogram_obj.Z_tree_format = mock_dendrogram  # Set the tree format
                mock_dendrogram_class.return_value = mock_dendrogram_obj

            # Make the request
            with open(test_image_path, 'rb') as f:
                response = client.post(
                    "/api/query",
                    files={"image": ("test_image.png", f, "image/png")},
                    data={
                        "current_model_id": query_request["current_model_id"],
                        "graph_type": query_request["graph_type"]
                    }
                )
            
            # Assert the response
            assert response.status_code == 201
            data = response.json()
            
            # Check response structure
            assert "query_result" in data
            assert "top_predictions" in data
            assert "image" in data
            
            # Verify the mocked data was returned
            assert data["query_result"] == verbal_explanation
            assert data["top_predictions"] == top_predictions
            
            # Verify the service functions were called correctly
            mock_query_predictions.assert_called_once_with(
                query_request["current_model_id"],
                query_request["graph_type"],
                MagicMock(),  # current_user
            )
            mock_query_model.assert_called_once_with(
                top_label,
                query_request["current_model_id"],
                query_request["graph_type"],
                MagicMock()  # current_user
            )
