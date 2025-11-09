# ==============================================================
# ENTRENAMIENTO DE MODELOS - PROYECTO DESPLIEGUE DE SOLUCIONES ANALÍTICAS
# Predicción del precio de viviendas en EE. UU. (Gradient Boosting)
# ==============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

# ==============================================================
# 1. CONEXIÓN A MLFLOW REMOTO (EC2)
# ==============================================================
mlflow.set_tracking_uri("http://3.214.199.14:5000")
mlflow.set_experiment("Gradient Boosting Regressor")

# ==============================================================
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ==============================================================
df = pd.read_csv("data/USAHousingDataset.csv")

# Limpieza y transformación básica
df = df.drop(columns=['date', 'street', 'statezip', 'country'])
df = df[df['price'] > 0]
df = pd.get_dummies(df, columns=['city'])

# Variables predictoras y objetivo
X = df.drop(columns=['price'])
y = np.log(df['price'])  # Escala logarítmica para estabilizar la varianza

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ==============================================================
# 3. DEFINICIÓN DE HIPERPARÁMETROS
# ==============================================================
param_grid = {
    "n_est": [100, 200, 300],
    "lr": [0.05, 0.1, 0.2],
    "max_d": [3, 5, 7]
}

# Generar combinaciones de hiperparámetros
param_combinations = list(itertools.product(
    param_grid["n_est"],
    param_grid["lr"],
    param_grid["max_d"]
))

# ==============================================================
# 4. ENTRENAMIENTO Y REGISTRO EN MLFLOW
# ==============================================================
for n_est, lr, max_d in param_combinations:
    nombre = f"GB_n{n_est}_lr{lr}_d{max_d}"

    with mlflow.start_run(run_name=nombre):
        modelo = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            random_state=42
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Métricas de evaluación
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Registro de parámetros y métricas
        mlflow.log_param("Modelo", nombre)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Evitar errores por caracteres inválidos
        safe_name = nombre.replace(".", "_").replace(":", "_").replace("/", "_")
        mlflow.sklearn.log_model(modelo, name=safe_name)

        print(f"{nombre} → RMSE: {rmse:.3f} | R²: {r2:.3f}")

print("\n Todos los modelos Gradient Boosting fueron registrados correctamente en MLflow remoto (EC2).")
