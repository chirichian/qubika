import mlflow
from qubika_challenge.globals import  MLFLOW_URI, EXPERIMENT
import logging

logging.basicConfig(level=logging.INFO)

def register_best_model(model_name:str, experiment_name:str):
    """Register the best model."""
    run_id = mlflow.search_runs(experiment_names=[experiment_name],
                        order_by=["metrics.f1_score DESC"]).iloc[0]["run_id"]
    logging.info(f"Registering the best model from run_id: {run_id}")
    
    mlflow.register_model(f"runs:/{run_id}/model", name=model_name)    
    
def register():
    """Register the best model."""
    experiment_name = EXPERIMENT.get("name")
    logging.info(f"Experiment name: {experiment_name}")
    model_name = EXPERIMENT.get("register_name")
    logging.info(f"Model name: {model_name}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    register_best_model(model_name, experiment_name)