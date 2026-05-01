import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformer:
    def __int__(self):
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transformer_object(self,train_path):
        try:
            df=pd.read_csv(train_path)
            num_colums=df.select_dtypes(include=['Float64','Int64']).columns
            print(num_colums)
            cat_columns=df.select_dtypes(exclude=['Float64','Int64']).columns
            numerical_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy='median')),
                       ("Scalar",StandardScaler())
                    
                    ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(stratergy='most_frequent')),
                    ("Encoder",OneHotEncoder())
                ]   
            )
            preprocessing=ColumnTransformer([("numerical_pipeline", numerical_pipeline,num_colums),
                                             ("cetegorical pipeline",categorical_pipeline,cat_columns)])
            logging.info('data transformer object is created')
            return preprocessing
        except Exception as e:
           raise CustomException(e,sys)
    
    def int_data_transform(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            preprocessor=self.get_data_transformer_object(train_path)
            traget_column='G3'
            numerical_features=df_train.select_dtypes(exclude=['object'])
            training_input_feature=df_train.drop(columns=[traget_column],axis=1)
            traing_target=df_train[traget_column]
            test_input_feature=df_test.drop(columns=[traget_column],axis=1)
            test_target=df_test[traget_column]
        except Exception as e:
            raise CustomException(e,sys)
