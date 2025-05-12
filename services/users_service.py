from utilss.classes.user import User
import json
import os

def initialize_user(email: str, password: str) -> User:
    """Initialize a new user."""
    user = User(user_id=None, email=email, password=password)
    user.create_user()
    return user

def mock_login(email: str, password: str) -> User:
    """
    Mock login function to simulate user authentication using users.json.
    """
    # Path to the users.json file
    USERS_PATH = os.getenv("USERS_PATH")
    users_json_path = os.path.join(USERS_PATH, "users.json")

    # Load users from the JSON file
    try:
        with open(users_json_path, "r") as file:
            users = json.load(file)
    except FileNotFoundError:
        return {"error": "Users database not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid users database format"}

    # Find the user by email
    user = next((u for u in users if u["email"] == email), None)
    if not user:
        return {"error": "User not found"}

    # Validate the password
    if user["password"] != password:
        return {"error": "Invalid password"}
    
    user_class = User(user_id=user['id'], email=user['email'], password=user['password'])
    user_class.load_models()
    return user_class

def get_current_session_user():
    return User(user_id="290924d2-6952-47f8-8308-7807d886e5b8", email="nurixhbh@gmail.com", password="123")