from fastapi.exceptions import RequestValidationError
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
from utilss import debug_utils

# Set up logging
logger = logging.getLogger(__name__)

logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3transfer").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for 422 validation errors.
    Constructs a custom error message based on the missing parameter.
    """
    logger.error(f"Validation error: {exc.errors()}")

    # Extract the missing parameter from the validation error details
    missing_params = []
    for error in exc.errors():
        if error["type"] == "missing":
            # Extract the parameter name from the error location
            loc = error.get("loc", [])
            if len(loc) > 1:  # Ensure the location has at least two elements
                param_name = loc[-1]  # Get the last element as the parameter name
                missing_params.append(param_name)
                print(f"Missing parameter: {param_name}")

    # Construct a custom error message
    if missing_params:
        detail = f"The required parameter(s) {', '.join(missing_params)} is/are missing."
    else:
        detail = "Validation error occurred."

    return JSONResponse(
        status_code=422,
        content={"detail": detail}
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTPException.
    Logs the error and returns a JSON response with the error details.
    """
    logger.error(f"HTTPException: {exc.detail} (status code: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """
    Custom handler for unexpected exceptions.
    Logs the error and returns a generic 500 error response.
    """
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )