import unittest
from unittest.mock import patch, mock_open
from utilss.classes.user import User
import os
import json
import uuid

class TestUser(unittest.TestCase):
    @patch("os.getenv")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_user(self, mock_open, mock_makedirs, mock_getenv):
        # Mock environment variable
        mock_getenv.return_value = "mock_users_path"

        # Create a User instance
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, email="test@example.com", password="password123")

        # Call create_user
        user.create_user()

        # Assert os.makedirs was called for the base directory and user-specific directories
        mock_makedirs.assert_any_call("mock_users_path", exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join("mock_users_path", user_id), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join("mock_users_path", user_id, "models"), exist_ok=True)

        # Assert the correct file operations were performed
        mock_open.assert_any_call(os.path.join("mock_users_path", "users.json"), "w")
        mock_open.assert_any_call(os.path.join("mock_users_path", user_id, "models.json"), "w")

        # Check the content written to users.json
        handle = mock_open()
        handle.write.assert_called()  # Ensure write was called
        written_data = json.loads(handle().write.call_args[0][0])
        self.assertEqual(written_data[0]["email"], "test@example.com")

    @patch("os.getenv")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_models(self, mock_open, mock_getenv):
        # Mock environment variable
        mock_getenv.return_value = "mock_users_path"

        # Mock models.json content
        mock_open.return_value.read.return_value = json.dumps({"models": ["model1", "model2"]})

        # Create a User instance
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, email="test@example.com", password="password123")

        # Call load_models
        user.load_models()

        # Assert models were loaded correctly
        self.assertEqual(user.get_models(), ["model1", "model2"])

if __name__ == "__main__":
    unittest.main()