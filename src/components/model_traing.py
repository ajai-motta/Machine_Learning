import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,RandomForestRegressor,GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from src.utils import evaluate_model
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    trained_mode_file_path=os.path.join('artifact','mymodel.pkl')


class ModelTriner:
    def __init__(self):
        self.model_triner_config=ModelTrainerConfig()
    
    def initate_model_train(self,train_array,test_arr,preprocesser_path):
        try:
            logging.info("model trining started")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],train_array[:,-1],test_arr[:,:-1],test_arr[:,-1],
            )
            models = {
                "linear_regression": LinearRegression(),
                "decision_tree": DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "knn": KNeighborsRegressor(),
                "catboost": CatBoostRegressor(verbose=0,train_dir=None),
               
            }
            model_report: dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model = max(model_report, key=lambda x: model_report[x]["r2_score"])
            logging.info(f"{best_model} with r2_score of {model_report[best_model]['r2_score']}") # dont use "" it says end of line
        except Exception as e:
            raise CustomException(e,sys)