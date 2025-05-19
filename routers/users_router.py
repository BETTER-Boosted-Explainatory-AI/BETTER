from fastapi import APIRouter, HTTPException, status, Request, Response, Cookie
from services.users_service import initialize_user
from request_models.users_model import UserCreateRequest
from services.auth_service import cognito_sign_up, cognito_login
from services.users_service import get_current_session_user

users_router = APIRouter()

## Login and register through our UI
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
    

## login and register through cognito UI
# @users_router.post(
#     "/cognito/callback",
#     status_code=status.HTTP_200_OK,
#     responses={
#         status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
#         status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
#     }
# )

# async def cognito_callback(request: Request, response: Response):
#     """
#     Receives Cognito tokens from frontend, verifies them, and processes user session.
#     """
#     try:
#         # Extract token from Authorization header
#         auth_header = request.headers.get("authorization")
#         if not auth_header or not auth_header.startswith("Bearer "):
#             raise HTTPException(status_code=400, detail="Authorization header missing or invalid")
#         token = auth_header.split("Bearer ")[1]

#         response.set_cookie(
#             key="session_token",
#             value=token,
#             httponly=True,      # Prevents JS access
#             secure=True,        # Only sent over HTTPS
#             samesite="lax",     # Adjust as needed
#             max_age=3600        # 1 hour, adjust as needed
#         )
        
#         # Verify the token and process user session
#         user = get_current_session_user(token)

#         if not user:
#             raise HTTPException(status_code=401, detail="Invalid token")
        
#         if not user.find_user_in_db():
#             new_user = initialize_user(id=user.user_id, email=user.email)
#             if not new_user:
#                 raise HTTPException(status_code=500, detail="Failed to create user in database")
#             print(f"User {user.user_id} created with email {user.email}")
#             return {"message": "User created successfully", "user_id": user.user_id}

        
#         return {"message": "Cognito callback processed successfully", "user": user}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# @users_router.get("/me")
# async def get_current_user(session_token: str = Cookie(None)):
#     if not session_token:
#         raise HTTPException(status_code=401, detail="Not authenticated")
#     user = get_current_session_user(session_token)
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid session token")
#     return {"user": user}