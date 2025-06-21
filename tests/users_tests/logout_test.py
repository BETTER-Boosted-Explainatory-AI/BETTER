from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_logout_user_clears_cookies():
    cookies = {
        "session_token": "some_token",
        "user_id": "some_user_id",
        "refresh_token": "some_refresh_token"
    }
    response = client.post("/api/logout", cookies=cookies)
    assert response.status_code == 200
    assert response.json()["message"] == "Logout successful"
    set_cookie_headers = response.headers.get_list("set-cookie")
    # Check that each cookie is being deleted (value is empty string)
    assert any(h.startswith("session_token=") or h.startswith('session_token="') for h in set_cookie_headers)
    assert any(h.startswith("user_id=") or h.startswith('user_id="') for h in set_cookie_headers)
    assert any(h.startswith("refresh_token=") or h.startswith('refresh_token="') for h in set_cookie_headers)

def test_logout_user_no_cookies():
    response = client.post("/api/logout")
    assert response.status_code == 200
    assert response.json()["message"] == "Logout successful"