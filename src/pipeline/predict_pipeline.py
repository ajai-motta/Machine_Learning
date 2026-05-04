import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_file
from src.components.data_transformation import DataTransformerConfig
from src.components.model_traing import ModelTrainerConfig
data_transformer_config=DataTransformerConfig()
model_trainer_config=ModelTrainerConfig()

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model=load_file(file_path=model_trainer_config.trained_model_file_path)
            preprocessor:ColumnTransformer=load_file(file_path=data_transformer_config.preprocessor_obj_file_path)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, school, sex, age, address, famsize, Pstatus,
                 Medu, Fedu, Mjob, Fjob, reason, guardian,
                 traveltime, studytime, failures, schoolsup, famsup,
                 paid, activities, nursery, higher, internet, romantic,
                 famrel, freetime, goout, Dalc, Walc, health,
                 absences, G1, G2):

        self.school = school
        self.sex = sex
        self.age = int(age)
        self.address = address
        self.famsize = famsize
        self.Pstatus = Pstatus
        self.Medu = int(Medu)
        self.Fedu = int(Fedu)
        self.Mjob = Mjob
        self.Fjob = Fjob
        self.reason = reason
        self.guardian = guardian
        self.traveltime = int(traveltime)
        self.studytime = int(studytime)
        self.failures = int(failures)
        self.schoolsup = schoolsup
        self.famsup = famsup
        self.paid = paid
        self.activities = activities
        self.nursery = nursery
        self.higher = higher
        self.internet = internet
        self.romantic = romantic
        self.famrel = int(famrel)
        self.freetime = int(freetime)
        self.goout = int(goout)
        self.Dalc = int(Dalc)
        self.Walc = int(Walc)
        self.health = int(health)
        self.absences = int(absences)
        self.G1 = int(G1)
        self.G2 = int(G2)
    def get_dta_as_data_frame(self):
        try:
            convet_dict={
                "school": [self.school],
                "sex": [self.sex],
                "age": [self.age],
                "address": [self.address],
                "famsize": [self.famsize],
                "Pstatus": [self.Pstatus],
                "Medu": [self.Medu],
                "Fedu": [self.Fedu],
                "Mjob": [self.Mjob],
                "Fjob": [self.Fjob],
                "reason": [self.reason],
                "guardian": [self.guardian],
                "traveltime": [self.traveltime],
                "studytime": [self.studytime],
                "failures": [self.failures],
                "schoolsup": [self.schoolsup],
                "famsup": [self.famsup],
                "paid": [self.paid],
                "activities": [self.activities],
                "nursery": [self.nursery],
                "higher": [self.higher],
                "internet": [self.internet],
                "romantic": [self.romantic],
                "famrel": [self.famrel],
                "freetime": [self.freetime],
                "goout": [self.goout],
                "Dalc": [self.Dalc],
                "Walc": [self.Walc],
                "health": [self.health],
                "absences": [self.absences],
                "G1": [self.G1],
                "G2": [self.G2],
            
            }
            logging.info('data too dta frame init')
            return pd.DataFrame(convet_dict)
        except Exception as e:
         raise CustomException(e,sys)
