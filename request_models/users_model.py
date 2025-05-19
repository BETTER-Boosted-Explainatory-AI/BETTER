from pydantic import BaseModel, EmailStr

class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str

class ConfirmUserRequest(BaseModel):
    id: str
    email: str
    confirmation_code: str