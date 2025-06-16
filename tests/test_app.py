import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure app.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = TestClient(app)

def test_search_success(monkeypatch):
    # Patch model and index for fast test
    response = client.post("/search", json={"query": "Diabetes", "top_k": 2})
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)

def test_search_empty_query():
    response = client.post("/search", json={"query": "", "top_k": 2})
    assert response.status_code == 400

def test_search_missing_query():
    response = client.post("/search", json={"top_k": 2})
    assert response.status_code == 422  # Unprocessable Entity (missing required field)