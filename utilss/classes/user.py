import os
import json
import uuid

class User:
    def __init__(self, user_id: uuid ,email: str, models: list = None):
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.email = email
        self.models = models if models is not None else []
        self.current_model = None
        self.USERS_PATH = os.getenv("USERS_PATH")
        self.users_json_path = os.path.join(self.USERS_PATH, "users.json")
        self.user_folder_path = os.path.join(self.USERS_PATH, self.user_id)
        self.models_json_path = os.path.join(self.user_folder_path, "models.json")
        self.current_model_json = os.path.join(self.user_folder_path, "current_model.json")

    def create_user(self):
        # Ensure the base directory exists
        os.makedirs(self.USERS_PATH, exist_ok=True)

        # Add user information to users.json
        user_data = {
            "id": self.user_id,
            "email": self.email,
            # "password": self.password  ## only for testing, will be hashed by aws incognito in the future
        }

        if os.path.exists(self.users_json_path):
            with open(self.users_json_path, "r") as file:
                users = json.load(file)
        else:
            users = []

        users.append(user_data)

        with open(self.users_json_path, "w") as file:
            json.dump(users, file, indent=4)

        # Create user folder and models.json
        os.makedirs(self.user_folder_path, exist_ok=True)

        with open(self.models_json_path, "w") as file:
            json.dump([], file, indent=4)

        with open(self.current_model_json, "w") as file:
            json.dump({}, file, indent=4)


    def load_models(self):
        if os.path.exists(self.models_json_path):
            with open(self.models_json_path, "r") as file:
                self.models = json.load(file)
        else:
            print(f"No models found for user {self.user_id}")

    def get_user_id(self):
        return self.user_id
    
    def get_models(self):
        return self.models
    
    def get_models_json_path(self):
        return self.models_json_path
    
    def add_model(self, model_info: dict):
        self.models.append(model_info)

        with open(self.models_json_path, "w") as file:
            json.dump([self.models], file, indent=4)

    def set_current_model(self, model_info: dict):
        self.current_model = model_info
        with open(self.current_model_json, "w") as file:
            json.dump(self.current_model, file, indent=4)
        return self.current_model
    
    def get_current_model(self):
        return self.current_model
    
    def get_user_folder(self):
        return self.user_folder_path
    
    def find_user_in_db(self):

        if os.path.exists(self.users_json_path):
            with open(self.users_json_path, "r") as file:
                users = json.load(file)
                for user in users:
                    if user["id"] == self.user_id:
                        return User(user_id=user["id"], email=user["email"])
        return None

        