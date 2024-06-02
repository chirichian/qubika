# Machine Learning Engineering Assignment

This assignment is expected to be completed by a Machine Learning Engineer within 3-4 hours.

### Assignment:

Your teammate Mike gave you a Machine Learning Model that needs to be deployed, examine the model.py file. 
Your task is to add the necessary components to get this model to production, 
this model is crucial for the company's success, so it should be available receive and predict data upon request.
This model should work for any cloud provider. Also this is the model v1 of a series of versions to be deployed.
(Optional): you can come up with a better model pipeline if you think the one provided is not suitable for this case.

Add all the necessary files, code, docs, to improve this repository and make this model ready to be shipped.

Note: You can reach out to us through email at marcos.soto@qubika.com or anibal.jasin@qubika.com but since
you could be working off hours we suggest you can write down questions and assumptions you came across
and what decision you take e.g. “I’m gonna assume this project should have X because of Y” or 
“I would ask about if there is some ACL for this model, so I would assume X”.

Some general considerations and guidance:

- It is expected that you include files, code, documentation, and any other resources you find useful in the new repository you create.
- A fully production-ready model deployed in the cloud is not required.
- You are not expected to spend money on cloud services, ensuring everything works locally is sufficient.
- It is acceptable if some aspects are not fully functional, our goal is to understand your approach to this problem.
- In case you don’t finish the assignment, you can add what other things you would do in the README.md.


---------------------------------------------------------------------------
version=0.0.3

# SOLUTION PROCESS
This is an approach of solution using mlflow to track experiments, comparing models and deploying the best model to production.

To set the enviroment poetry was used, to install poetry run the following command:
    
```pip install Poetry==1.6.1```

To install the dependencies run the following command:

```poetry install```

Before starting the process, run the following command to start mlflow server locally:

```poetry run mlflow ui --port=5050```

In our case, we are going to run MlFlow in the port 5050 locally.
If you want to change the port, you can change the port in the command above. But remember to update MLFLOW_URI in the globals.py file.

# TRAIN process
To train a modelo first you have to configure some globals variables in the file ```globals.py```.

```name```: refers to the name of the experiment.
```run_name```: refers to the name of the run, you can have many runs in the same experiment, and then compare all and get the best one to deploy.
```params```: refers to the hyperparameters of the model.
```register_name```: refers to the name of the model used to register in mlflow.

After that you can run the following command to train the model:

```poetry run train```

Mlflow will track the experiment and you can see the results in the mlflow interface, like all the metrics, hyperparameters, and the model that was saved in the folder ```qubika_challenge/models```.


# REGISTER the best model
After training the model, you can register the best model in the mlflow interface. If you did different experiments, you can compare all of them and choose the best one to register. Once you chose the experiment where is the best model, you have to update de variable EXPERIMENT in the globals.py file with the experiment name of the best experiment. 


To register the model, you have to run the following command:

```poetry run register```

Each time you register your model in mlflow each model will have a version, you can see the version of the model in the mlflow interface. This will help you to deploy the model in the future, you can choose the VERSION of the model that you want to deploy and update in the globals.py file.


# BATCH PREDICTION
To get predictions in batch, you have to upload the file in qubika_challenge/data/predict/predictions.csv, you can choose the name of the file, but the file has to be in csv format. After that, you update the filename in globals.


Running the following command:

```poetry run batch_prediction```

you will get the predictions in the file ```qubika_challenge/data/predict/predictions.csv```
**Remember that you can change the name or location's files**

# Next steps
- Create a script to deploy the model in the cloud.
- Create a script to get predictions in real-time.
- Create a script to monitor the model in production. Saving the input and output of the model.
- Create a monitor to check the data quality, using quantity of null values, distribution of income, etc  
