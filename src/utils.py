import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path,obj):
    try:
       dir_path=os.path.dirname(file_path)
       os.makedirs(dir_path,exist_ok=True)
       with open(file_path,"+wb") as file:
            dill.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}

        for name, model in models.items():
            # Train model
            print(model)
            model.fit(x_train, y_train)

            # Predictions
            y_pred = model.predict(x_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store results
            report[name] = {
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_file(file_path):
    try:
        with open(file_path,'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)