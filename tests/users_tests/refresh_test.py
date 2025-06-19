import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)

# Mock refresh_session result
def mock_refresh_session_success(request, user_id):
    return {
        "AuthenticationResult": {
            "IdToken": "new_id_token",
            "RefreshToken": "new_refresh_token"
        }
    }

def mock_refresh_session_no_refresh_token(request, user_id):
    return {
        "AuthenticationResult": {
            "IdToken": "new_id_token",
            "RefreshToken": None
        }
    }

def mock_refresh_session_fail(request, user_id):
    return None

@patch("app.users_router.refresh_session", side_effect=mock_refresh_session_success)
def test_refresh_user_session_success(mock_refresh):
    cookies = {"user_id": "test-user-id"}
    response = client.post("/api/refresh", cookies=cookies)
    assert response.status_code == 200
    assert response.json()["message"] == "Session refreshed successfully"
    assert "session_token" in response.cookies
    assert response.cookies["session_token"] == "new_id_token"
    assert "refresh_token" in response.cookies
    assert response.cookies["refresh_token"] == "new_refresh_token"

@patch("app.users_router.refresh_session", side_effect=mock_refresh_session_no_refresh_token)
def test_refresh_user_session_no_refresh_token(mock_refresh):
    cookies = {"user_id": "test-user-id"}
    response = client.post("/api/refresh", cookies=cookies)
    assert response.status_code == 200
    assert response.json()["message"] == "Session refreshed successfully"
    assert "session_token" in response.cookies
    assert response.cookies["session_token"] == "new_id_token"
    assert "refresh_token" not in response.cookies

def test_refresh_user_session_no_user_id_cookie():
    response = client.post("/api/refresh")
    assert response.status_code == 401
    assert response.json()["detail"] == "user_id cookie not found"

@patch("app.users_router.refresh_session", side_effect=mock_refresh_session_fail)
def test_refresh_user_session_refresh_fail(mock_refresh):
    cookies = {"user_id": "test-user-id"}
    response = client.post("/api/refresh", cookies=cookies)
    assert response.status_code == 401
    assert response.json()["detail"] == "Failed to refresh session"

@patch("app.users_router.refresh_session", side_effect=Exception("Unexpected error"))
def test_refresh_user_session_internal_error(mock_refresh):
    cookies = {"user_id": "test-user-id"}
    response = client.post("/api/refresh", cookies=cookies)
    assert response.status_code == 500
    assert "Unexpected error" in response.json()["detail"]