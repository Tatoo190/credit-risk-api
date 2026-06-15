"""
Entrenamiento del modelo de scoring de riesgo crediticio.
Registra experimentos con MLflow: métricas, parámetros y el modelo.
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import joblib

from preprocess import load_data, clean_data, split_features_target


def train_model(data_path: str = "../data/cs-training.csv"):
    # 1. Cargar y limpiar datos
    df = load_data(data_path)
    df = clean_data(df)
    X, y = split_features_target(df)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Configurar MLflow
    mlflow.set_experiment("credit-risk-scoring")

    with mlflow.start_run():
        # Hiperparámetros del modelo
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "class_weight": "balanced",  # importante: dataset desbalanceado (pocos defaults)
        }

        # 4. Entrenar modelo
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # 5. Predecir y evaluar
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 6. Loggear en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 7. Loggear el modelo en MLflow
        mlflow.sklearn.log_model(model, "model")

        # 8. Guardar también localmente para la API (joblib)
        joblib.dump(model, "./api/model.pkl")

        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

        # 9. Validar umbral mínimo de calidad (gate para CI/CD)
        MIN_AUC = 0.70
        if auc < MIN_AUC:
            raise ValueError(
                f"Modelo no cumple el umbral mínimo de AUC ({MIN_AUC}). AUC obtenido: {auc:.4f}"
            )

        return model, auc


if __name__ == "__main__":
    train_model()