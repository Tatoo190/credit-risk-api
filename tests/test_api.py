"""
Tests de la API de scoring crediticio.
Verifica que los endpoints respondan correctamente y validen inputs.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "api"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """El endpoint raíz debe responder con status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_check():
    """El health check debe confirmar que el modelo está cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_valid_input():
    """Una predicción con datos válidos debe devolver probabilidad y nivel de riesgo."""
    payload = {
        "revolving_utilization": 0.5,
        "age": 35,
        "n_30_59_days_late": 0,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "n_open_credit_lines": 5,
        "n_90_days_late": 0,
        "n_real_estate_loans": 1,
        "n_60_89_days_late": 0,
        "n_dependents": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert 0 <= data["default_probability"] <= 1
    assert data["risk_level"] in ["BAJO", "MEDIO", "ALTO"]
    assert "timestamp" in data


def test_predict_high_risk_profile():
    """Un perfil con mucha mora e ingreso bajo debe clasificar como riesgo más alto
    que un perfil sin mora e ingreso alto."""
    high_risk_payload = {
        "revolving_utilization": 0.95,
        "age": 25,
        "n_30_59_days_late": 5,
        "debt_ratio": 0.9,
        "monthly_income": 1000,
        "n_open_credit_lines": 10,
        "n_90_days_late": 3,
        "n_real_estate_loans": 0,
        "n_60_89_days_late": 2,
        "n_dependents": 4,
    }
    low_risk_payload = {
        "revolving_utilization": 0.1,
        "age": 45,
        "n_30_59_days_late": 0,
        "debt_ratio": 0.1,
        "monthly_income": 10000,
        "n_open_credit_lines": 3,
        "n_90_days_late": 0,
        "n_real_estate_loans": 1,
        "n_60_89_days_late": 0,
        "n_dependents": 0,
    }

    high_risk_response = client.post("/predict", json=high_risk_payload)
    low_risk_response = client.post("/predict", json=low_risk_payload)

    high_proba = high_risk_response.json()["default_probability"]
    low_proba = low_risk_response.json()["default_probability"]

    assert high_proba > low_proba


def test_predict_invalid_input_negative_age():
    """Edad negativa debe ser rechazada por validación de Pydantic (422)."""
    payload = {
        "revolving_utilization": 0.5,
        "age": -5,
        "n_30_59_days_late": 0,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "n_open_credit_lines": 5,
        "n_90_days_late": 0,
        "n_real_estate_loans": 1,
        "n_60_89_days_late": 0,
        "n_dependents": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    """Si falta un campo requerido, debe devolver error 422."""
    payload = {
        "revolving_utilization": 0.5,
        "age": 35,
        # falta el resto de campos
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422