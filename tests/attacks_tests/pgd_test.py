import sys
import os
import pytest
import requests
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app 

# Create test client
client = TestClient(app)

def test_pgd_attack_endpoint():
    """Test the PGD attack endpoint with default parameters."""
    
    payload = {
        "dataset_name": "cifar100",
        "image_index": 0,
        "epsilon": 0.1,
        "alpha": 0.01,
        "num_steps": 10,  # Fewer steps for faster testing
        "targeted": False,
        "target_class": None,
        "threshold": None
    }
    
    # Send the request
    response = client.post("/attack/pgd", json=payload)
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Parse the response body
    response_data = response.json()
    
    # Assert the response structure
    assert "status" in response_data
    assert response_data["status"] == "success"
    assert "data" in response_data
    
    # Assert the attack results
    result = response_data["data"]
    assert result["attack_type"] == "PGD"
    assert "original_class" in result
    assert "adversarial_class" in result
    assert "attack_success" in result
    assert "original_score" in result
    assert "adversarial_score" in result
    assert "detection_success" in result
    assert "l2_norm" in result
    assert "linf_norm" in result
    assert "execution_time" in result
    assert "parameters" in result


def test_pgd_attack_with_targeted():
    """Test the PGD attack endpoint with targeted attack."""
    
    # Define the request payload for targeted attack
    payload = {
        "dataset_name": "cifar100",
        "image_index": 0,
        "epsilon": 0.1,
        "alpha": 0.01,
        "num_steps": 10,  # Fewer steps for faster testing
        "targeted": True,
        "target_class": 5,  # Target class index
        "threshold": None
    }
    
    # Send the request
    response = client.post("/attack/pgd", json=payload)
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Parse the response body
    response_data = response.json()
    
    # Assert the targeted attack results
    result = response_data["data"]
    assert "targeted_attack" in result
    assert result["targeted_attack"]["target_class"] == 5


def test_pgd_attack_with_invalid_dataset():
    """Test the PGD attack endpoint with an invalid dataset name."""
    
    # Define the request payload with invalid dataset
    payload = {
        "dataset_name": "invalid_dataset",
        "image_index": 0,
        "epsilon": 0.1,
        "alpha": 0.01,
        "num_steps": 10
    }
    
    # Send the request
    response = client.post("/attack/pgd", json=payload)
    
    # Assert the response status code for bad request
    assert response.status_code == 400
    
    # Parse the response body
    response_data = response.json()
    
    # Assert the error message
    assert "detail" in response_data
    assert "Invalid dataset" in response_data["detail"]


def test_get_cluster_types():
    """Test the endpoint for getting cluster types."""
    
    # Send the request
    response = client.get("/attack/pgd/types")
    
    # Assert the response status code
    assert response.status_code == 200
    
    # Parse the response body
    response_data = response.json()
    
    # Assert the response structure
    assert "status" in response_data
    assert response_data["status"] == "success"
    assert "data" in response_data
    
    # Assert the cluster types
    cluster_types = response_data["data"]
    assert "SIMILARITY" in cluster_types
    assert "DISSIMILARITY" in cluster_types
    assert "CONFUSION_MATRIX" in cluster_types


class TestPGDAttackWithPostman:
    """Manual test instructions for testing with Postman."""
    
    def test_pgd_attack_with_postman(self):
        """
        This test provides instructions for manual testing with Postman.
        
        Steps to test with Postman:
        1. Start your FastAPI server (e.g., uvicorn main:app --reload)
        2. Open Postman
        3. Create a new POST request to http://localhost:8000/attack/pgd
        4. Set the Content-Type header to application/json
        5. Add the following JSON payload to the request body:
        
        {
            "dataset_name": "cifar100",
            "image_index": 0,
            "epsilon": 0.1,
            "alpha": 0.01,
            "num_steps": 40,
            "targeted": false,
            "target_class": null,
            "threshold": null
        }
        
        6. Send the request and verify the response
        
        For targeted attack, use this payload:
        
        {
            "dataset_name": "cifar100",
            "image_index": 0,
            "epsilon": 0.1,
            "alpha": 0.01,
            "num_steps": 40,
            "targeted": true,
            "target_class": 5,
            "threshold": null
        }
        
        7. To get available cluster types, create a GET request to http://localhost:8000/attack/pgd/types
        """
        pass  # This is not an actual test, just documentation


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])