from pydantic import BaseModel, EmailStr

class UserCreateRequest(BaseModel):
    email: EmailStr
    password: str