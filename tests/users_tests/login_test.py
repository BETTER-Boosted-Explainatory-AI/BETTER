from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)

def mock_cognito_login_success(user_create_request):
    return {
        'AuthenticationResult': {
            'AccessToken': 'fake-access-token',
            'ExpiresIn': 3600,
            'TokenType': 'Bearer'
        }
    }

def mock_cognito_login_invalid_credentials(user_create_request):
    raise Exception("An error occurred (NotAuthorizedException) when calling the InitiateAuth operation: Incorrect username or password")

def mock_cognito_login_user_not_found(user_create_request):
    raise Exception("An error occurred (UserNotFoundException) when calling the InitiateAuth operation: User does not exist")

def mock_cognito_validation_error(user_create_request):
    raise Exception("The required parameter(s) [Password] were not specified or are empty")

def mock_cognito_unknown_error(user_create_request):
    raise Exception("An unknown error occurred")

@patch('services.auth_service.cognito_login', side_effect=mock_cognito_login_success)
def test_login_user_success(mock_login):
    payload = {
        "email": "nurixhbh@gmail.com",
        "password": "Moran2604!@#"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Login successful"

@patch('services.auth_service.cognito_login', side_effect=mock_cognito_login_invalid_credentials)
def test_login_user_invalid_credentials(mock_login):
    payload = {
        "email": "nurixhbh@gmail.com",
        "password": "wrongpassword"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


@patch('services.auth_service.cognito_login', side_effect=mock_cognito_validation_error)
def test_login_user_validation_error(mock_login):
    payload = {
        "email": "nurixhbh@gmail.com"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 422
    assert "The required parameter(s)" in response.json()["detail"]

@patch('services.auth_service.cognito_login', side_effect=mock_cognito_unknown_error)
def test_login_user_missing_password(mock_login):
    payload = {
        "email": "nurixhbh@gmail.com",
        "password": ""
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 500
    assert "An unknown error occurred" in response.json()["detail"]

def test_login_user_unknown_error():
    payload = {
        "email": "",
        "password": "SomePassword123!"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == "Validation error occurred."

@patch('services.auth_service.cognito_login', side_effect=mock_cognito_validation_error)
def test_login_user_missing_email(mock_login):
    payload = {
        "password": "SomePassword123!"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 422
    assert "The required parameter(s)" in response.json()["detail"]

@patch('services.auth_service.cognito_login', side_effect=mock_cognito_login_user_not_found)
def test_login_user_invalid_email_format(mock_login):
    payload = {
        "email": "not-an-email",
        "password": "SomePassword123!"
    }
    response = client.post("/api/login", json=payload)
    assert response.status_code == 422