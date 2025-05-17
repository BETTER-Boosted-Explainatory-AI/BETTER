import os
import json
import uuid

class User:
    def __init__(self, user_id: uuid ,email: str, password: str, models: list = None):
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.email = email
        self.password = password ## only for testing, will be hashed by aws incognito in the future
        self.models = models if models is not None else []
        self.current_model = None

    def create_user(self):
        # Logic to create a new user in the database
        USERS_PATH = os.getenv("USERS_PATH")

        # Define paths
        users_json_path = os.path.join(USERS_PATH, "users.json")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        models_json_path = os.path.join(user_folder_path, "models.json")
        current_model_json = os.path.join(user_folder_path, "current_model.json")

        # Ensure the base directory exists
        os.makedirs(USERS_PATH, exist_ok=True)

        # Add user information to users.json
        user_data = {
            "id": self.user_id,
            "email": self.email,
            "password": self.password  ## only for testing, will be hashed by aws incognito in the future
        }

        if os.path.exists(users_json_path):
            with open(users_json_path, "r") as file:
                users = json.load(file)
        else:
            users = []

        users.append(user_data)

        with open(users_json_path, "w") as file:
            json.dump(users, file, indent=4)

        # Create user folder and models.json
        os.makedirs(user_folder_path, exist_ok=True)

        with open(models_json_path, "w") as file:
            json.dump([], file, indent=4)

        with open(current_model_json, "w") as file:
            json.dump({}, file, indent=4)


    def load_models(self):
        # Logic to load models from the database
        USERS_PATH = os.getenv("USERS_PATH")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        models_json_path = os.path.join(user_folder_path, "models.json")

        if os.path.exists(models_json_path):
            with open(models_json_path, "r") as file:
                # data = json.load(file)
                self.models = json.load(file)
        else:
            print(f"No models found for user {self.user_id}")

    def get_user_id(self):
        return self.user_id
    
    def get_models(self):
        return self.models
    
    def get_models_json_path(self):
        USERS_PATH = os.getenv("USERS_PATH")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        models_json_path = os.path.join(user_folder_path, "models.json")
        return models_json_path
    
    def add_model(self, model_info: dict):
        self.models.append(model_info)
        USERS_PATH = os.getenv("USERS_PATH")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        models_json_path = os.path.join(user_folder_path, "models.json")

        with open(models_json_path, "w") as file:
            json.dump([self.models], file, indent=4)

    def set_current_model(self, model_info: dict):
        self.current_model = model_info
        USERS_PATH = os.getenv("USERS_PATH")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        current_model_json = os.path.join(user_folder_path, "current_model.json")
        with open(current_model_json, "w") as file:
            json.dump(self.current_model, file, indent=4)
        return self.current_model
    
    def get_current_model(self):
        return self.current_model
    
    def get_user_folder(self):
        USERS_PATH = os.getenv("USERS_PATH")
        user_folder_path = os.path.join(USERS_PATH, self.user_id)
        return user_folder_path

        