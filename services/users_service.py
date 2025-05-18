from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, Depends
from services.auth_service import verify_cognito_jwt
from utilss.classes.user import User

def initialize_user(id, email) -> User:
    """Initialize a new user."""
    user = User(user_id=id, email=email)
    user.create_user()
    return user

security = HTTPBearer()

def get_current_session_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = verify_cognito_jwt(token)
        user_id = payload.get("sub")
        email = payload.get("email")
        print(f"Active user: {user_id} with email {email}")
        return User(user_id=user_id, email=email)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")
