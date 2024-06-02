"Global variables for the project"

#MLFLOW
MLFLOW_URI="http://127.0.0.1:8080/"

#Transformer and train data
NUMERIC_FEATURES=['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']
TARGET_COL='Loan_Approval'

#Experiment
EXPERIMENT = {"name": "Qubika",
              "run_name":"LR_1",
              "params":{"intercept_scaling": 1, "penalty": "l2", "max_iter":100},
              "register_name":"QubikaModel"
              }

VERSION = "2"

#Data Source
PATH_PREDICT="qubika_challenge/data/predict/"
PATH_TRAIN="qubika_challenge/data/raw/"
FILE_NAME_TRAIN="dataset.csv"
FILE_NAME_PREDICT="dataset.csv"
PREDICTIONS_OUTPUT = "predictions.csv"



