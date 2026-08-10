import sys
import os
import json
import tempfile


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import mlflow
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np

from config.constants import Constants
from config.experiment_config import ExperimentConfig



def log_snapshots(snapshots: dict, prefix: str = "snapshots"):
    with tempfile.TemporaryDirectory() as temp_dir:
        def save_item(name, value):
            path = os.path.join(temp_dir, name)
            if isinstance(value, pd.DataFrame):
                value.to_csv(f"{path}.csv", index=True)
            elif isinstance(value, np.ndarray):
                np.save(f"{path}.npy", value)
            elif isinstance(value, dict):
                os.makedirs(path, exist_ok=True)
                for k, v in value.items():
                    save_item(f"{name}/{k}", v)
            else:
                with open(f"{path}.json", "w") as f:
                    json.dump(value, f)
        
        for k, v in snapshots.items():
            save_item(k, v)
        
        mlflow.log_artifacts(temp_dir, artifact_path=prefix)


def log_experiment(
        config: ExperimentConfig, 
        validator_snapshots: dict[str, object]|None=None,
        history: dict[str, object]|None=None): 
    # mlflow server --host 127.0.0.1 --port 8080
    
    mlflow.set_tracking_uri(uri=Constants.MLFlow.TRACKING_URI)

    mlflow.set_experiment(f'{config.observability_config.experiment_name}')

    # Start an MLflow run
    with mlflow.start_run():
        if config.raw_config is None:
            raise ValueError(
                "Experiment config has no serializable source manifest"
            )

        # Keep the full resolved config as an artifact and only searchable,
        # scalar values as MLflow parameters.
        mlflow.log_dict(config.raw_config, "config/train_config.json")
        mlflow.log_params({
            "schema_version": config.raw_config["schema_version"],
            "config_revision": config.raw_config["config_revision"],
            "model": config.raw_config["model"]["name"],
            "feature_count": len(config.raw_config["data"]["features"]),
            "symbol_count": len(config.raw_config["data"]["symbols"]),
            "batch_size": config.raw_config["train"]["batch_size"],
            "learning_rate": config.raw_config["train"]["optimizer"]
                .get("params", {})
                .get("lr"),
        })

        if validator_snapshots is not None:
            log_snapshots(validator_snapshots, prefix="validator_snapshots")

        if history is not None:
            for epoch in range(config.train_config.num_epochs): 
                mlflow.log_metrics(
                    { metric_name: metric_values[epoch] for metric_name, metric_values in history.items() },
                    step=epoch)

        # if model is not None:
        #     mlflow.pytorch.log_model(
        #         pytorch_model=model,
        #         # signature=infer_signature(input_data_sample, model(input_data_sample)),
        #         # input_example=input_data_sample.cpu().numpy(),
        #         registered_model_name=config.model_config.registered_model_name,
        #     )