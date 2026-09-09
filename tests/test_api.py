from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_classify_valid_input():
    payload = {
        "num_functional_requirements": 5, "num_integrations": 2,
        "requires_web_component": 1, "requires_mobile_component": 0,
        "requires_ai_ml_component": 0, "requires_cloud_infra": 0,
        "requires_data_pipeline": 0, "num_technologies_mentioned": 3,
        "has_deadline": 1, "deadline_days": 30,
        "complexity_keyword_count": 1, "requirement_text_length": 200,
    }
    r = client.post("/classify", json=payload)
    assert r.status_code == 200
    assert "project_category" in r.json()

def test_classify_invalid_input():
    r = client.post("/classify", json={"num_functional_requirements": -1})
    assert r.status_code == 422
    
