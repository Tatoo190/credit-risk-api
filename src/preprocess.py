"""
Módulo de preprocesamiento de datos para el modelo de riesgo crediticio.
Dataset: "Give Me Some Credit" (Kaggle) - predicción de default a 2 años.
"""
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset crudo desde un CSV."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset:
    - Renombra columnas a snake_case
    - Maneja valores nulos en ingresos y número de dependientes
    - Elimina outliers extremos en variables de edad y morosidad
    """
    df = df.copy()

    # Renombrar columnas a snake_case para consistencia
    df.columns = [
        "id",
        "default_flag",
        "revolving_utilization",
        "age",
        "n_30_59_days_late",
        "debt_ratio",
        "monthly_income",
        "n_open_credit_lines",
        "n_90_days_late",
        "n_real_estate_loans",
        "n_60_89_days_late",
        "n_dependents",
    ]

    # Imputar nulos: ingreso mensual con la mediana
    df["monthly_income"] = df["monthly_income"].fillna(
        df["monthly_income"].median()
    )

    # Imputar nulos: dependientes con 0 (asumimos sin dependientes si no se reporta)
    df["n_dependents"] = df["n_dependents"].fillna(0)

    # Eliminar outliers de edad (edades imposibles)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]

    # Eliminar outliers extremos en mora (valores como 96, 98 son códigos de error)
    mora_cols = ["n_30_59_days_late", "n_60_89_days_late", "n_90_days_late"]
    for col in mora_cols:
        df = df[df[col] < 90]

    # Quitar columna id, no aporta al modelo
    df = df.drop(columns=["id"])

    return df


def split_features_target(df: pd.DataFrame):
    """Separa features (X) y target (y)."""
    X = df.drop(columns=["default_flag"])
    y = df["default_flag"]
    return X, y


if __name__ == "__main__":
    df_raw = load_data("data/cs-training.csv")
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/cleaned_data.csv", index=False)
    print(f"Datos limpios guardados. Shape final: {df_clean.shape}")