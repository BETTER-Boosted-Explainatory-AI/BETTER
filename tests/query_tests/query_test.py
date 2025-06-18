import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
from utilss.classes.user import User
import uuid
from pathlib import Path
import tensorflow as tf

from app import app
from services.users_service import require_authenticated_user
from .query_mock_data import user, models_metadata, model_id, verbal_explanation, top_label,  top_k_predictions, labels, mock_dendrogram

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


def local_model_loader(*args, **kwargs):
    # Path to your local test model file
    return tf.keras.models.load_model("tests/query_tests/resnet50_imagenet.keras")


def test_query_success(client, query_request, test_image_path):
    with patch("services.models_service.Dendrogram") as mock_dendrogram_class, \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.s3_file_exists", return_value=True), \
            patch("services.models_service.read_json_from_s3", return_value=[models_metadata['models'][0]]), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch("services.dataset_service.get_dataset_labels", return_value=labels), \
            patch("services.models_service.get_top_k_predictions", return_value=top_k_predictions), \
            patch("services.models_service.get_user_models_info", return_value=models_metadata['models'][0]):
        with patch("routers.query_router.query_model", return_value=verbal_explanation), \
                patch("routers.query_router.query_predictions", return_value=(top_label, top_k_predictions)), \
                patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
            # Setup Dendrogram mock
            mock_dendrogram_obj = MagicMock()
            mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj
            mock_dendrogram_obj.Z_tree_format = mock_dendrogram
            mock_dendrogram_class.return_value = mock_dendrogram_obj

            with open(test_image_path, 'rb') as f:
                response = client.post(
                    "/api/query",
                    files={"image": ("test_image.png", open(query_request["image"], "rb"), "image/png")},
                    data={
                        "current_model_id": query_request["current_model_id"],
                        "graph_type": query_request["graph_type"]
                    }
                )
                assert response.status_code == 201
                data = response.json()
                assert "query_result" in data
                assert "top_predictions" in data
                assert "image" in data
                assert data["query_result"] == verbal_explanation
                assert data["top_predictions"] == [
                    list(x) for x in top_k_predictions]


def test_query_model_not_found(client, query_request):
    with patch("services.models_service.Dendrogram") as mock_dendrogram_class, \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.s3_file_exists", return_value=False), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
        # Setup Dendrogram mock
        mock_dendrogram_obj = MagicMock()
        mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj
        mock_dendrogram_obj.Z_tree_format = mock_dendrogram
        mock_dendrogram_class.return_value = mock_dendrogram_obj
        response = client.post(
            "/api/query",
            files={"image": ("test_image.png", open(query_request["image"], "rb"), "image/png")},
            data={
                "current_model_id": query_request["current_model_id"],
                "graph_type": query_request["graph_type"]
            }
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Model file does not exist in S3"}


def test_query_invalid_image(client, query_request):
    with patch("services.models_service.Dendrogram") as mock_dendrogram_class, \
            patch("services.models_service.load_model_from_s3", return_value=MagicMock()), \
            patch("services.models_service.s3_file_exists", return_value=True), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
            
            # Setup Dendrogram mock
            mock_dendrogram_obj = MagicMock()
            mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj
            mock_dendrogram_obj.Z_tree_format = mock_dendrogram
            mock_dendrogram_class.return_value = mock_dendrogram_obj
            response = client.post(
                "/api/query",
                files={"image": ("invalid_image.txt",
                                "This is not an image", "text/plain")},
                data={
                    "current_model_id": query_request["current_model_id"],
                    "graph_type": query_request["graph_type"]
                }
            )
            assert response.status_code == 422
            assert response.json() == {"detail": "Uploaded file is not a valid image."}

def test_query_internal_server_error(client, query_request):
    with patch("services.models_service.Dendrogram") as mock_dendrogram_class, \
            patch("routers.query_router.query_predictions", side_effect=Exception("Internal Server Error")), \
            patch("services.models_service.s3_file_exists", return_value=True), \
            patch("services.models_service.get_users_s3_client", return_value=MagicMock()), \
            patch("services.models_service.get_model_specific_file", return_value="test/path"), \
            patch.dict("os.environ", {"S3_USERS_BUCKET_NAME": "test-bucket"}):
        
        # Setup Dendrogram mock
        mock_dendrogram_obj = MagicMock()
        mock_dendrogram_obj.load_dendrogram.return_value = mock_dendrogram_obj
        mock_dendrogram_obj.Z_tree_format = mock_dendrogram
        mock_dendrogram_class.return_value = mock_dendrogram_obj
        
        response = client.post(
            "/api/query",
            files={"image": ("test_image.png", open(query_request["image"], "rb"), "image/png")},
            data={
                "current_model_id": query_request["current_model_id"],
                "graph_type": query_request["graph_type"]
            }
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}
