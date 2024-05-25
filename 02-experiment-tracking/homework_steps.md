## Q3. Train a model with autolog

- Step 1 -> Import MLflow + initial settings

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

- Step 2 -> wrap the training using `mlflow.start_run()` and add `mlflow.autolog()`
```
    with mlflow.start_run():

        mlflow.autolog()

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        mlflow.log_metric("rmse", rmse)
```
- Step 3 -> Open MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
