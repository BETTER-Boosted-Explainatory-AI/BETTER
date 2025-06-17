from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)

def mock_cognito_sign_up(user_create_request):
    return {'UserSub': 'fake-user-id'}

def mock_cognito_sign_up_username_exists(user_create_request):
    raise Exception("An error occurred (UsernameExistsException) when calling the SignUp operation: User already exists")

def mock_cognito_sign_up_weak_password(user_create_request):
    raise Exception("An error occurred (InvalidPasswordException) when calling the SignUp operation: Password not long enough")

def mock_cognito_sign_up_validation_error(user_create_request):
    raise Exception("The required parameter(s) [Password] were not specified or are empty")

@patch('services.auth_service.cognito_sign_up', side_effect=mock_cognito_sign_up)
def test_register_user_success(mock_signup):
    payload = {
        "email": "test10@example.com",
        "password": "TestPassword123!"
    }
    response = client.post("/api/register", json=payload)
    assert response.status_code == 201
    assert response.json()["message"] == "User created successfully"
    assert response.json()["user"]["email"] == "test10@example.com"

@patch('services.auth_service.cognito_sign_up', side_effect=mock_cognito_sign_up_username_exists)
def test_user_already_exists(mock_signup):
    payload = {
        "email": "test1@example.com",
        "password": "TestPassword123!"
    }
    response = client.post("/api/register", json=payload)
    assert response.status_code == 500
    assert "UsernameExistsException" in response.json()["detail"]
    assert "User already exists" in response.json()["detail"]

@patch('services.auth_service.cognito_sign_up', side_effect=mock_cognito_sign_up_validation_error)
def test_register_user_validation_error(mock_signup):
    payload = {
        "email": "test4@example.com"
    }
    response = client.post("/api/register", json=payload)
    assert response.status_code == 422
    assert "The required parameter(s)" in response.json()["detail"]

@patch('services.auth_service.cognito_sign_up', side_effect=mock_cognito_sign_up_weak_password)
def test_weak_password(mock_signup):
    payload = {
        "email": "test3@example.com",
        "password": "weak"
    }
    response = client.post("/api/register", json=payload)
    assert response.status_code == 500
    assert "InvalidPasswordException" in response.json()["detail"]
    assert "Password not long enough" in response.json()["detail"]