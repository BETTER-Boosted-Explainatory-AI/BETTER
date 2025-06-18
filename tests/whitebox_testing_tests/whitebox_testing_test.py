import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import moto
from utilss.classes.user import User
import uuid
from pathlib import Path
import io
from PIL import Image
import numpy as np

from app import app
from services.users_service import require_authenticated_user
from .wbt_mock_data import user, models_metadata, model_id, source_labels, target_labels, mock_edges_df
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
def fake_edges_csv():
    # Create a fake CSV in memory
    csv_content = mock_edges_df
    return csv_content


@pytest.fixture
def mock_s3(fake_edges_csv):
    mock_client = MagicMock()
    mock_client.head_object.return_value = {}
    mock_client.get_object.return_value = {
        'Body': io.BytesIO(fake_edges_csv.encode('utf-8'))
    }
    with patch("utilss.s3_utils.get_users_s3_client", return_value=mock_client), \
            patch("services.whitebox_testing_service.get_users_s3_client", return_value=mock_client), \
            patch("utilss.classes.edges_dataframe.get_users_s3_client", return_value=mock_client):
        yield


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
def mock_dataset(test_image_path):
    # Load the test image as a numpy array
    img = Image.open(test_image_path).convert("RGB")
    img_array = np.array(img)

    class DummyDataset:
        def get_train_image_by_id(self, image_id):
            # Return the image array and a dummy label
            return img_array, "Persian_cat"

    with patch("services.dataset_service._load_dataset", return_value=DummyDataset()):
        yield


@pytest.fixture
def mock_whitebox_testing_request():
    return {
        "model_id": model_id,
        "graph_type": "similarity",
        "source_labels": source_labels,
        "target_labels": target_labels,
    }


def test_whitebox_testing_success(client, mock_whitebox_testing_request, mock_s3, mock_dataset, fake_edges_csv):
    # Create a fake DataFrame to return from EdgesDataframe.get_dataframe
    fake_df = pd.read_csv(io.StringIO(fake_edges_csv))

    # Mock the models_metadata to return the expected metadata and patch S3/model/dataframe functions
    with patch("services.models_service.get_user_models_info", return_value=models_metadata['models']), \
            patch("services.whitebox_testing_service._get_model_filename", return_value="fake/model/path"), \
            patch("utilss.classes.edges_dataframe.EdgesDataframe.load_dataframe", return_value=None), \
            patch("utilss.classes.edges_dataframe.EdgesDataframe.get_dataframe", return_value=fake_df), \
            patch("utilss.classes.edges_dataframe.EdgesDataframe._s3_file_exists", return_value=True), \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.read_json_from_s3", return_value=[models_metadata['models'][0]]), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch("services.models_service.get_user_models_info", return_value=models_metadata['models'][0]), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
        response = client.post(
            "/api/whitebox_testing",
            json=mock_whitebox_testing_request
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all("image" in item for item in data)
        assert all("image_id" in item for item in data)
        assert all("matches" in item for item in data)


def test_whitebox_testing_model_not_found(client, mock_whitebox_testing_request, mock_s3):
    # Use a valid but non-existent UUID for model_id
    mock_whitebox_testing_request["model_id"] = "123e4567-e89b-12d3-a456-426614174000"

    # Patch get_user_models_info to return an empty list (model not found)
    with patch("services.models_service.get_user_models_info", return_value=[]):
        response = client.post(
            "/api/whitebox_testing",
            json={
                "model_id": model_id,
                "graph_type": "similarity",
                "source_labels": source_labels,
            }
        )
        assert response.status_code == 422


def test_whitebox_testing_missing_parameters(client, mock_whitebox_testing_request, mock_s3, mock_dataset):
    with patch("utilss.classes.edges_dataframe.EdgesDataframe._s3_file_exists", return_value=True), \
        patch("services.models_service.get_user_models_info", return_value=models_metadata['models']), \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.read_json_from_s3", return_value=[models_metadata['models'][0]]), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch("services.models_service.get_user_models_info", return_value=models_metadata['models'][0]), \
            patch("utilss.classes.edges_dataframe.EdgesDataframe.get_dataframe", side_effect=Exception("Simulated internal error")), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
        response = client.post(
            "/api/whitebox_testing",
            json=mock_whitebox_testing_request
        )
        assert response.status_code == 500


def test_whitebox_testing_internal_error(client, mock_whitebox_testing_request, mock_s3, mock_dataset):
    with patch("utilss.classes.edges_dataframe.EdgesDataframe._s3_file_exists", return_value=True), \
        patch("services.models_service.get_user_models_info", return_value=models_metadata['models']), \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.read_json_from_s3", return_value=[models_metadata['models'][0]]), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch("services.models_service.get_user_models_info", return_value=models_metadata['models'][0]), \
            patch("utilss.classes.edges_dataframe.EdgesDataframe.get_dataframe", side_effect=Exception("Simulated internal error")), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
        response = client.post(
            "/api/whitebox_testing",
            json=mock_whitebox_testing_request
        )
        assert response.status_code == 500
