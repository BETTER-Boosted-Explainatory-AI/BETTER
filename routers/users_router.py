from fastapi import APIRouter, HTTPException, status, Response, Request, Depends
from services.users_service import initialize_user
from request_models.users_model import UserCreateRequest, ConfirmUserRequest
from services.auth_service import cognito_sign_up, cognito_login, confirm_user_signup, refresh_session
from services.users_service import get_current_session_user, require_authenticated_user
from utilss.classes.user import User
from botocore.exceptions import ClientError

users_router = APIRouter()

@users_router.post(
    "/api/register",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
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
        user_dict = {"id": user_id, "email": email}
        return {"message": "User created successfully", "user": user_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@users_router.post(
    "/api/confirm",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
    }
)
def confirm_user(user_confirm_request: ConfirmUserRequest) -> dict:
    """
    Confirm a user's registration with the code sent to their email.
    """
    try:
        response = confirm_user_signup(
            user_confirm_request.email, user_confirm_request.confirmation_code)
        if not response:
            raise HTTPException(
                status_code=400, detail="Invalid confirmation code or email")
        user = initialize_user(id=user_confirm_request.id,
                               email=user_confirm_request.email)
        if not user:
            raise HTTPException(
                status_code=500, detail="Failed to create user in database")
        return {"message": "User confirmed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@users_router.post(
    "/api/login",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
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
        refresh_token = auth_header.get("RefreshToken")

        response.set_cookie(
            key="session_token",
            value=id_token,
            httponly=True,      # Prevents JS access
            secure=False,      # Only for local testing, set to True in production!
            # secure=True,        # Only sent over HTTPS
            # samesite="none",     # Adjust as needed
            samesite="lax",      # Adjust as needed
            max_age=900        # 1 hour, adjust as needed
        )

        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,      # Prevents JS access
            secure=False,      # Only for local testing, set to True in production!
            # secure=True,        # Only sent over HTTPS
            # samesite="none",     # Adjust as needed
            samesite="lax",      # Adjust as needed
            max_age=7*24*3600      # 7 days, adjust as needed
        )

        user = get_current_session_user(id_token)
        user_dict = {"id": user.user_id, "email": user.email}

        response.set_cookie(
            key="user_id",
            value=user.user_id,
            httponly=True,      # Prevents JS access
            secure=False,      # Only for local testing, set to True in production!
            # secure=True,        # Only sent over HTTPS
            # samesite="none",     # Adjust as needed
            samesite="lax",      # Adjust as needed
            max_age=7*24*3600      # 7 days, adjust as needed
        )

        return {"message": "Login successful", "user": user_dict}
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == "NotAuthorizedException":
            raise HTTPException(
                status_code=401, detail="Incorrect username or password")
        elif error_code == "UserNotFoundException":
            raise HTTPException(status_code=404, detail="User not found")
        else:
            raise HTTPException(
                status_code=500, detail="An unknown error occurred")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@users_router.get(
    "/api/me",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
    }
)
def get_active_user_info(current_user: User = Depends(require_authenticated_user)):
    user_dict = {"id": current_user.user_id, "email": current_user.email}
    return {"message": "User information retrieved successfully", "user": user_dict}


@users_router.post(
    "/api/refresh",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
    }
)
def refresh_user_session(request: Request, response: Response):
    """
    Refresh the user's session by generating a new session token.
    """
    try:
        user_id = request.cookies.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="user_id cookie not found")

        refresh_result = refresh_session(request, user_id)
        if not refresh_result:
            raise HTTPException(
                status_code=401, detail="Failed to refresh session")
        auth_header = refresh_result.get("AuthenticationResult")
        id_token = auth_header.get("IdToken")
        refresh_token = auth_header.get("RefreshToken")

        # Set new cookies
        response.set_cookie(
            key="session_token",
            value=id_token,
            httponly=True,
            # secure=True,  # Set to True in production!
            secure=False,  # Only for local testing, set to True in production!
            # samesite="none",
            samesite="lax",  # Adjust as needed
            max_age=900
        )

        if refresh_token:
            response.set_cookie(
                key="refresh_token",
                value=refresh_token,
                httponly=True,
                # secure=True,  # Set to True in production!
                secure=False,  # Only for local testing, set to True in production!
                # samesite="none",
                samesite="lax",  # Adjust as needed
                max_age=7*24*3600  # 7 days
            )

        return {"message": "Session refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@users_router.post(
    "/api/logout",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Logout successful"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"}
    }
)
def logout_user(response: Response):
    """
    Log out the user by clearing the session cookie.
    """
    response.delete_cookie(key="session_token")
    response.delete_cookie(key="user_id")
    response.delete_cookie(key="refresh_token")

    return {"message": "Logout successful"}
