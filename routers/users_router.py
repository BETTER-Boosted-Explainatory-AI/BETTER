from fastapi import APIRouter, HTTPException, status
from services.users_service import initialize_user
from request_models.users_model import UserCreateRequest
from services.auth_service import cognito_sign_up, cognito_login

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
        cognito_user = cognito_sign_up(user_create_request)
        user_id = cognito_user['UserSub']
        email = user_create_request.email
        print(f"User {user_id} created with email {email}")
        user = initialize_user(id=user_id, email=email)
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
        response = cognito_login(user_create_request)
        return {"message": "Login successful", "auth_result": response.get("AuthenticationResult")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))