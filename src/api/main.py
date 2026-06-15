"""
API de scoring de riesgo crediticio.
Expone un endpoint /predict que recibe datos de un cliente y devuelve
la probabilidad de default.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging
from datetime import datetime

# Configurar logging para monitoreo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API para predecir probabilidad de default crediticio",
    version="1.0.0",
)

# Cargar modelo al iniciar la app
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


class ClientData(BaseModel):
    """Esquema de entrada: datos del cliente para scoring."""

    revolving_utilization: float = Field(
        ..., description="Ratio de uso de líneas de crédito revolventes", ge=0
    )
    age: int = Field(..., description="Edad del cliente", ge=18, le=100)
    n_30_59_days_late: int = Field(
        ..., description="Veces con mora de 30-59 días", ge=0
    )
    debt_ratio: float = Field(..., description="Ratio de endeudamiento", ge=0)
    monthly_income: float = Field(..., description="Ingreso mensual", ge=0)
    n_open_credit_lines: int = Field(
        ..., description="Número de líneas de crédito abiertas", ge=0
    )
    n_90_days_late: int = Field(..., description="Veces con mora de 90+ días", ge=0)
    n_real_estate_loans: int = Field(
        ..., description="Número de préstamos hipotecarios", ge=0
    )
    n_60_89_days_late: int = Field(
        ..., description="Veces con mora de 60-89 días", ge=0
    )
    n_dependents: int = Field(..., description="Número de dependientes", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Esquema de salida: resultado del scoring."""

    default_probability: float
    risk_level: str
    timestamp: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Credit Risk Scoring API"}


@app.get("/health")
def health_check():
    """Endpoint de health check para monitoreo y orquestadores (k8s, etc)."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    """
    Recibe los datos de un cliente y devuelve la probabilidad de default
    junto con un nivel de riesgo categórico.
    """
    try:
        # Convertir input a DataFrame con el orden correcto de columnas
        input_df = pd.DataFrame([client.model_dump()])

        # Predecir probabilidad
        proba = model.predict_proba(input_df)[0, 1]

        # Clasificar nivel de riesgo
        if proba < 0.2:
            risk_level = "BAJO"
        elif proba < 0.5:
            risk_level = "MEDIO"
        else:
            risk_level = "ALTO"

        response = PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Log para monitoreo/trazabilidad
        logger.info(
            f"Predicción realizada: proba={response.default_probability}, "
            f"risk_level={response.risk_level}"
        )

        return response

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))