from fastapi import APIRouter, HTTPException, status, Response, Depends
from services.users_service import initialize_user
from request_models.users_model import UserCreateRequest, ConfirmUserRequest
from services.auth_service import cognito_sign_up, cognito_login, confirm_user_signup
from services.users_service import get_current_session_user, require_authenticated_user
from utilss.classes.user import User
from botocore.exceptions import ClientError

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
        user_dict = {"id":user_id, "email": email}
        return {"message": "User created successfully", "user": user_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@users_router.post(
    "/confirm",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
def confirm_user(user_confirm_request: ConfirmUserRequest) -> dict:
    """
    Confirm a user's registration with the code sent to their email.
    """
    try:
        response = confirm_user_signup(user_confirm_request.email, user_confirm_request.confirmation_code)
        if not response:
            raise HTTPException(status_code=400, detail="Invalid confirmation code or email")
        user = initialize_user(id=user_confirm_request.id, email=user_confirm_request.email)
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create user in database")
        return {"message": "User confirmed successfully"}
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
def login_user(user_create_request: UserCreateRequest, response: Response) -> dict:
    """
    Mock login function to simulate user authentication.
    """
    try:
        cognito_response = cognito_login(user_create_request)
        if not cognito_response:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        auth_header = cognito_response.get("AuthenticationResult")
        id_token = auth_header.get("IdToken")

        response.set_cookie(
            key="session_token",
            value=id_token,
            httponly=True,      # Prevents JS access
            secure=False,      # Only for local testing, set to True in production!
            # secure=True,        # Only sent over HTTPS
            samesite="lax",     # Adjust as needed
            max_age=3600        # 1 hour, adjust as needed
        )

        user = get_current_session_user(id_token)
        user_dict = {"id":user.user_id, "email": user.email}

        return {"message": "Login successful", "user": user_dict}
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == "NotAuthorizedException":
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        elif error_code == "UserNotFoundException":
            raise HTTPException(status_code=404, detail="User not found")
        else:
            raise HTTPException(status_code=500, detail="An unknown error occurred")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@users_router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"}
    }
)
def get_active_user_info(current_user: User = Depends(require_authenticated_user)):
    user_dict = {"id":current_user.user_id, "email": current_user.email}
    return {"message": "User information retrieved successfully", "user": user_dict}
    
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