import os
import uuid
import pandas as pd
import numpy as np
import yaml
import click
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn

print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
client = mlflow.tracking.MlflowClient()

class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, data):
        predictions = self.model.predict(data)
        data.insert(0, "prediction", predictions.tolist())
        out_dir = os.environ.get("MLFLOW_MONITORING_DIR","out")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir,str(uuid.uuid4())+".csv")
        with open(path, "w") as f:
            data.to_csv(f, index=False)
        return predictions 

def build_data(data_path):
    col_label = "quality"
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=2019)
    X_train = train.drop([col_label], axis=1)
    X_test = test.drop([col_label], axis=1)
    y_train = train[[col_label]]
    y_test = test[[col_label]]
    return X_train, X_test, y_train, y_test 

def train(data_path, max_depth, max_leaf_nodes):
    X_train, X_test, y_train, y_test = build_data(data_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("  run_id:", run_id)
        print("  experiment_id:", experiment_id)
        print("  experiment_name:", client.get_experiment(experiment_id).name)

        # Create model
        dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        print("Model:\n ", dt)

        # Fit and predict
        dt.fit(X_train, y_train)
        predictions = dt.predict(X_test)

        # MLflow params
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

        # MLflow metrics
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, predictions)))
        mlflow.log_metric("r2", r2_score(y_test, predictions))
        mlflow.log_metric("mae",  mean_absolute_error(y_test, predictions))
        
        # MLflow tags
        mlflow.set_tag("data_path", data_path)
        mlflow.set_tag("mlflow_version", mlflow.__version__)

        # Pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([('step', dt)])

        # Log model
        mlflow.sklearn.log_model(pipeline, "sklearn-model")
        register_model(run_id, "sklearn-model", "sklearn-monitor")

        # Log custom model
        path = "conda_custom.yaml"
        with open(path, "r") as f:
            dct = yaml.safe_load(f)
        mlflow.pyfunc.log_model("sklearn-model-custom", python_model=CustomModel(pipeline), conda_env=dct)
        register_model(run_id, "sklearn-model-custom", "sklearn-monitor-custom")

    return (experiment_id,run_id)

def register_model(run_id, artifact_model_name, registered_model_name):
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_model_name}", name=registered_model_name)
    client.transition_model_version_stage(
        name=version.name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True)

@click.command()
@click.option("--experiment_name", help="Experiment name", type=str, default="sklearn_monitor")
@click.option("--data-path", help="Data path", type=str, default="data/wine-quality-white.csv")
@click.option("--max-depth", help="Max depth", type=int, default=None)
@click.option("--max-leaf-nodes", help="Max leaf nodes", type=int, default=None)
def main(experiment_name, data_path, max_depth, max_leaf_nodes):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    mlflow.set_experiment(experiment_name)
    _,run_id =  train(data_path, max_depth, max_leaf_nodes)

if __name__ == "__main__":
    main()
