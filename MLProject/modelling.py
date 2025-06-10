import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("ci_workflow_experiment")

# Load data
X_train = pd.read_csv("cropclimate_preprocessing/X_train.csv")
X_test = pd.read_csv("cropclimate_preprocessing/X_test.csv")
y_train = pd.read_csv("cropclimate_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("cropclimate_preprocessing/y_test.csv").values.ravel()

# Training
mlflow.autolog()
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    input_example = X_test.iloc[:5]
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")
