import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import moto
from utilss.classes.user import User
import uuid
from pathlib import Path

from app import app
from services.users_service import require_authenticated_user