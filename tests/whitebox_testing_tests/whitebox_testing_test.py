import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
from utilss.classes.user import User
import uuid
from pathlib import Path

from app import app
from services.users_service import require_authenticated_user

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