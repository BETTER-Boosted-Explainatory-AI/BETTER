from fastapi import APIRouter, HTTPException, status
from services.users_service import initialize_user, mock_login
from request_models.users_model import UserCreateRequest

users_router = APIRouter()

@users_router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
def register_user(user_create_request: UserCreateRequest) -> dict:
    """
    Register a new user.
    """
    try:
        email = user_create_request.email
        password = user_create_request.password
        print(f"Creating user with email: {email}")
        user = initialize_user(email=email, password=password)
        print(f"User {user.user_id} created with email {user.email}")
        return {"message": "User created successfully", "user_id": user.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@users_router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
def login_user(user_create_request: UserCreateRequest):
    """
    Mock login function to simulate user authentication.
    """
    try:
        email = user_create_request.email
        password = user_create_request.password
        print(f"Logging in user with email: {email}")
        user = mock_login(email=email, password=password)
        print(f"User {user.user_id} logged in successfully")
        return {"message": "Login successful", "user_id": user.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))