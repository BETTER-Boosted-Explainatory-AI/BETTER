import hmac
import hashlib
import base64
import os
import boto3
import requests
from jose import jwt, JWTError

cognito_client_secret = os.getenv("COGNITO_CLIENT_SECRET")
cognito_client_id = os.getenv("COGNITO_CLIENT_ID")
aws_region = os.getenv("AWS_REGION")
cognito_user_pool_id = os.getenv("COGNITO_USER_POOL_ID")
incognito_client = boto3.client('cognito-idp', region_name=aws_region)

def get_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    message = username + client_id
    dig = hmac.new(
        client_secret.encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

def cognito_sign_up(user_create_request):
    """
    Sign up a new user in AWS Cognito.
    """
    # Generate the secret hash

    secret_hash = get_secret_hash(user_create_request.email, cognito_client_id, cognito_client_secret)

    return incognito_client.sign_up(
                ClientId=os.getenv("COGNITO_CLIENT_ID"),
                Username=user_create_request.email,
                Password=user_create_request.password,
                SecretHash=secret_hash
            )

def cognito_login(user_create_request):
    """
    Log in a user using AWS Cognito.
    """
    # Generate the secret hash
    secret_hash = get_secret_hash(user_create_request.email, cognito_client_id, cognito_client_secret)

    return incognito_client.initiate_auth(
                ClientId=cognito_client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': user_create_request.email,
                    'PASSWORD': user_create_request.password,
                    'SECRET_HASH': secret_hash
                }
            )

def confirm_user_signup(email: str, confirmation_code: str):
    """
    Confirm a user's signup using the code sent to their email.
    """

    secret_hash = get_secret_hash(email, cognito_client_id, cognito_client_secret)
    
    response = incognito_client.confirm_sign_up(
        ClientId=cognito_client_id,
        Username=email,
        ConfirmationCode=confirmation_code,
        SecretHash=secret_hash
    )
    return response

# Download and cache the JWKS (public keys)
JWKS_URL = f"https://cognito-idp.{aws_region}.amazonaws.com/{cognito_user_pool_id}/.well-known/jwks.json"
jwks = requests.get(JWKS_URL).json()

def verify_cognito_jwt(token: str):
    try:
        # Decode and verify the JWT
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=cognito_client_id,
            issuer=f"https://cognito-idp.{aws_region}.amazonaws.com/{cognito_user_pool_id}",
            options={"verify_at_hash": False}
        )
        return payload
    except JWTError as e:
        raise Exception(f"Token verification failed: {str(e)}")