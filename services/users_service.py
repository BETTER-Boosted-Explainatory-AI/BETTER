from fastapi import Cookie, HTTPException
from services.auth_service import verify_cognito_jwt
from utilss.classes.user import User

def initialize_user(id, email) -> User:
    """Initialize a new user."""
    user = User(user_id=id, email=email)
    user.create_user()
    return user

def get_current_session_user(token: str):
    try:
        payload = verify_cognito_jwt(token)
        user_id = payload.get("sub")
        email = payload.get("email")
        print(f"Active user: {user_id} with email {email}")
        return User(user_id=user_id, email=email)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

def require_authenticated_user(session_token: str = Cookie(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        user = get_current_session_user(session_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session token")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return user
