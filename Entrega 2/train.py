import mlflow

# Conectar al servidor remoto MLflow (EC2)
mlflow.set_tracking_uri("http://18.212.99.123:5000")

# Crear o usar un experimento específico
mlflow.set_experiment("USA-House-Prices")


import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 1. CARGA DE DATOS
df = pd.read_csv("data/USAHousingDataset.csv")

# Preprocesamiento igual al del notebook
df = df.drop(columns=['date', 'street', 'statezip', 'country'])
df = df[df['price'] > 0]
df = pd.get_dummies(df, columns=['city'])

X = df.drop(columns=['price'])
y = np.log(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 2. CONFIGURACIÓN MLFLOW
mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
mlflow.set_experiment("USA-House-Prices")

# 3. MODELOS
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=0)
}

# 4. LOOP DE ENTRENAMIENTO Y TRACKING
for nombre, modelo in modelos.items():
    with mlflow.start_run(run_name=nombre):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_param("model", nombre)

        mlflow.sklearn.log_model(modelo, artifact_path="model")

        print(f"{nombre} → RMSE: {rmse:.3f} | R²: {r2:.3f}")

print("\n Todos los modelos fueron registrados en MLflow.")
