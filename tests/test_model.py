"""
Tests del modelo entrenado: verifica artefactos y calidad mínima.
"""
import sys
import os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_model_file_exists():
    """El artefacto del modelo debe existir tras el entrenamiento."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "api", "model.pkl"
    )
    assert os.path.exists(model_path), "model.pkl no existe. Ejecuta train.py primero."


def test_model_predicts_probability():
    """El modelo debe poder predecir y devolver una probabilidad válida (0-1)."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "api", "model.pkl"
    )
    model = joblib.load(model_path)

    sample_input = pd.DataFrame([{
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
    }])

    proba = model.predict_proba(sample_input)[0, 1]
    assert 0 <= proba <= 1