import mlflow
from qubika_challenge.utils import read_file, save_file
from qubika_challenge.globals import MLFLOW_URI, EXPERIMENT, VERSION, PATH_PREDICT, TARGET_COL, PREDICTIONS_OUTPUT, FILE_NAME_PREDICT
import logging

logging.basicConfig(level=logging.INFO)


def load_model(model_name:str, version:str):
    """Load the model."""
    logging.info("Loading the model")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    logging.info("Model loaded")
    return model

def predict(model):
    """Get batch predictions."""
    logging.info("Predicting the target variable")
    data  = read_file(PATH_PREDICT, FILE_NAME_PREDICT)
    
    data[TARGET_COL]= data[TARGET_COL] = model.predict(data)
    logging.info("Predictions completed")
    #save file as csv in the same directory
    save_file(data, PATH_PREDICT, PREDICTIONS_OUTPUT)
    logging.info(f"Predictions saved in {PATH_PREDICT}")

def predict_batch():
    mlflow.set_tracking_uri(MLFLOW_URI)
    model= load_model(EXPERIMENT.get("register_name"),VERSION)
    predict(model)