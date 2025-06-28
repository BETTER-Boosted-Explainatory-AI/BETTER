import unittest
from unittest.mock import patch, mock_open, MagicMock
from utilss.classes.user import User
import os
import json
import uuid

class TestUser(unittest.TestCase):
    @patch("os.getenv")
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_user(self, mock_open, mock_path_exists, mock_makedirs, mock_getenv):
        mock_getenv.return_value = "mock_users_path"
        def path_exists_side_effect(path):
            if path.endswith("users.json"):
                return False  # users.json doesn't exist
            return True  # Directories exist
            
        mock_path_exists.side_effect = path_exists_side_effect
        mock_file_data = {}
        def mock_open_side_effect(filename, mode="r", *args, **kwargs):
            m = mock_open()
            if filename.endswith("users.json") and mode == "r":
                m.return_value.read.return_value = "[]" 
            return m()
            
        mock_open.side_effect = mock_open_side_effect
        
        # Create a User instance
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, email="test@example.com", password="password123")
        
        # Call create_user
        user.create_user()
        
        mock_makedirs.assert_any_call("mock_users_path", exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join("mock_users_path", user_id), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join("mock_users_path", user_id, "models"), exist_ok=True)
        
        mock_open.assert_any_call(os.path.join("mock_users_path", "users.json"), "w")
        mock_open.assert_any_call(os.path.join("mock_users_path", user_id, "models.json"), "w")

    @patch("os.getenv")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_models(self, mock_open, mock_path_exists, mock_getenv):
        mock_getenv.return_value = "mock_users_path"
        mock_path_exists.return_value = True
        mock_open.return_value.read.return_value = json.dumps({"models": ["model1", "model2"]})
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, email="test@example.com", password="password123")
        user.load_models()
        self.assertEqual(user.get_models(), ["model1", "model2"])

if __name__ == "__main__":
    unittest.main()