from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)

# Mock user object
class MockUser:
    user_id = "test-user-id"
    email = "test@example.com"
  
@patch('app.get_current_session_user', return_value=MockUser())
def test_get_active_user_info(mock_get_user):
    response = client.get("/api/me", cookies={"session_token": "faketoken"})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "User information retrieved successfully"
    assert data["user"]["id"] == "test-user-id"
    assert data["user"]["email"] == "test@example.com"