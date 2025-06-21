from unittest.mock import MagicMock, patch
import requests
import os
import dotenv 
dotenv.load_dotenv()

# os.environ.setdefault("AWS_REGION", "us-east-1")
# os.environ.setdefault("COGNITO_USER_POOL_ID", "local-pool")
# os.environ.setdefault("COGNITO_CLIENT_ID", "local-client")

_fake_resp = MagicMock()
_fake_resp.json.return_value = {"keys": []}          
patcher = patch.object(requests, "get", return_value=_fake_resp)
patcher.start()

import uuid
from typing import List

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from routers.model_router import model_router                
from services.users_service import require_authenticated_user         

# the route file did:  from services.models_service import get_user_models_info
MODEL_ROUTE_MODULE = "routers.model_router"                  

# ── helpers ────────────────────────────────────────────────────────
def _sample_models() -> List[dict]:
    """Return two mock model records (one with a succeeded job)."""
    return [
        {
            "model_id": "123",
            "file_name": "model_a.keras",
            "dataset": "cifar100",
            "graph_type": ["similarity"],
            "batch_jobs": [
                {"job_graph_type": "similarity", "job_status": "succeeded"},
                {"job_graph_type": "dissimilarity", "job_status": "failed"},
            ],
        },
        {
            "model_id": "456",
            "file_name": "model_b.keras",
            "dataset": "imagenet",
            "graph_type": ["dissimilarity"],
            "batch_jobs": [
                {"job_graph_type": "dissimilarity", "job_status": "running"}
            ],
        },
    ]


def _fake_user() -> MagicMock:
    """Return a stub User object."""
    user = MagicMock()
    user.user_id = str(uuid.uuid4())
    user.email = "alice@example.com"
    # current model info (mutable so PUT can update it)
    user._current_model = {
        "model_id": "123",
        "file_name": "model_a.keras",
        "dataset": "cifar100",
        "graph_type": "similarity",
    }

    user.get_current_model.side_effect = lambda: user._current_model

    def _set_current_model(info: dict):
        user._current_model = info
        return user._current_model

    user.set_current_model.side_effect = _set_current_model
    user.get_user_folder.return_value = user.user_id
    return user


# ── pytest fixtures ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app():
    app = FastAPI()
    app.include_router(model_router)
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _override_auth_dependency(app):
    """
    Always inject the fake user instead of running Cognito verification.
    """
    user = _fake_user()
    app.dependency_overrides[require_authenticated_user] = lambda: user
    yield
    app.dependency_overrides.clear()


# ── tests ──────────────────────────────────────────────────────────
@patch(f"{MODEL_ROUTE_MODULE}.get_user_models_info")
def test_get_models_returns_all(mock_get_info, client):
    mock_get_info.return_value = _sample_models()

    res = client.get("/api/models")

    assert res.status_code == status.HTTP_200_OK
    payload = res.json()
    assert len(payload) == 2
    assert {m["model_id"] for m in payload} == {"123", "456"}


@patch(f"{MODEL_ROUTE_MODULE}.get_user_models_info")
def test_get_models_filter_succeeded(mock_get_info, client):
    mock_get_info.return_value = _sample_models()

    res = client.get("/api/models", params={"status": "succeeded"})

    assert res.status_code == status.HTTP_200_OK
    payload = res.json()

    # Only the first mock model has a succeeded job
    assert len(payload) == 1
    model = payload[0]
    assert model["model_id"] == "123"
    # graph_type is rewritten to the succeeded job(s) only
    assert set(model["graph_type"]) == {"similarity"}


def test_get_current_model(client):
    res = client.get("/api/models/current")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["model_id"] == "123"
    assert data["graph_type"] == "similarity"


@patch(f"{MODEL_ROUTE_MODULE}.get_user_models_info")
def test_put_current_model_updates_graph_type(mock_get_info, client):
    mock_get_info.return_value = {
        "model_id": "123",
        "file_name": "model_a.keras",
        "dataset": "cifar100",
        "graph_type": ["similarity", "dissimilarity"],
        "batch_jobs": [],
    }

    res = client.put(
        "/api/models/current",
        json={"model_id": "123", "graph_type": "dissimilarity"},
    )

    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["model_id"] == "123"
    assert data["graph_type"] == "dissimilarity"

    # A second GET should reflect the update
    res2 = client.get("/api/models/current")
    assert res2.json()["graph_type"] == "dissimilarity"
