from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)

# Mock user object
class MockUser:
    user_id = "02f58494-a0c1-7003-7f34-2e1be083c8fa"
    email = "ixMomozz@gmail.com"

# Test: successful retrieval of user info with valid session
@patch('services.auth_service.cognito_login', return_value=MagicMock(get="faketoken"))
@patch('services.users_service.get_current_session_user', return_value=MockUser())
def test_get_active_user_info(mock_get_user, mock_cognito_login):
    response = client.get("/api/me", cookies={"session_token": "faketoken"})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "User information retrieved successfully"
    assert data["user"]["id"] == "02f58494-a0c1-7003-7f34-2e1be083c8fa"
    assert data["user"]["email"] == "ixMomozz@gmail.com"

# Test: missing session token
def test_get_user_info_no_session():
    response = client.get("/api/me")
    assert response.status_code == 401 or response.status_code == 403

# Test: invalid session token
@patch('services.auth_service.cognito_login', return_value=MagicMock(get="invalidtoken"))
@patch('services.users_service.get_current_session_user', side_effect=Exception("Invalid session"))
def test_get_user_info_invalid_session(mock_get_user, mock_cognito_login):
    response = client.get("/api/me", cookies={"session_token": "invalidtoken"})
    assert response.status_code == 401 or response.status_code == 403

# Test: user not found in session
@patch('services.auth_service.cognito_login', return_value=MagicMock(get="faketoken"))
@patch('services.users_service.get_current_session_user', return_value=None)
def test_get_user_info_user_not_found(mock_get_user, mock_cognito_login):
    response = client.get("/api/me", cookies={"session_token": "faketoken"})
    assert response.status_code == 404 or response.status_code == 401 or response.status_code == 403