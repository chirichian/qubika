import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from qubika_challenge.transformer import preprocessor
from qubika_challenge.utils import read_file
from qubika_challenge.globals import NUMERIC_FEATURES, PATH_TRAIN, TARGET_COL, FILE_NAME_TRAIN, EXPERIMENT, MLFLOW_URI
import logging

logging.basicConfig(level=logging.INFO)

def training():
    """Train the model."""
    # Get the global variables
    run_name = EXPERIMENT.get("run_name")
    logging.info(f"Run name: {run_name}")
    params = EXPERIMENT.get("params")
    logging.info(f"Params: {params}")
    experiment_name = EXPERIMENT.get("name")
    logging.info(f"Experiment name: {experiment_name}")

    #get preprocecssor pipeline
    logging.info("Defining the preprocessor")
    processor = preprocessor(NUMERIC_FEATURES)
    #load data
    logging.info(f"Loading the data from {PATH_TRAIN}")
    df = read_file(PATH_TRAIN, FILE_NAME_TRAIN)
    logging.info(f"df shape: {df.shape}")
    # Start an MLflow run
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    train(df, TARGET_COL, params, run_name, processor)
    
def train(df,target_col, params, run_name, processor):
    # Splitting the data into features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]  
    # Starting an MLflow run
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=run_name):
        # Training the Logistic Regression model
        model = LogisticRegression(**params)
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        #create execution pipeline
        clf = Pipeline(
        steps=[("preprocessor", processor), ("classifier", model)])
        #train model
        clf.fit(X_train, y_train)
        #get predictions
        y_pred = clf.predict(X_test)
        mlflow.end_run()
