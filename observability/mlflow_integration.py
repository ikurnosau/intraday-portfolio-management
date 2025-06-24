import sys
import os

import mlflow.pytorch
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import mlflow
from mlflow.models import infer_signature
import time
import torch

from config.constants import Constants
from config.experiment_config import ExperimentConfig, ObservabilityConfig
from config.config_serialization import serialize_config


def log_experiment(
        config: ExperimentConfig, 
        model: torch.nn,
        history: dict[str, object],
        input_data_sample: torch.Tensor): 
    # mlflow server --host 127.0.0.1 --port 8080
    
    mlflow.set_tracking_uri(uri=Constants.MLFlow.TRACKING_URI)

    mlflow.set_experiment(f'{config.observability_config.experiment_name}')

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(serialize_config(config))

        # Log the loss metric
        for epoch in range(config.train_config.num_epochs): 
            mlflow.log_metrics(
                { metric_name: metric_values[epoch] for metric_name, metric_values in history.items() },
                step=epoch)

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            # signature=infer_signature(input_data_sample, model(input_data_sample)),
            # input_example=input_data_sample.cpu().numpy(),
            registered_model_name=config.model_config.registered_model_name,
        )

        # # Set a tag that we can use to remind ourselves what this model was for
        # mlflow.set_logged_model_tags(
        #     model_info.model_id, {"Training Info": config.model_config}
        # )