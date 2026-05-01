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
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transformer_object(self,train_path):
        try:
            df=pd.read_csv(train_path)
            target_column = "G3"
            X = df.drop(columns=[target_column])
            num_colums=X.select_dtypes(include=['Float64','Int64']).columns
            print(num_colums)
            cat_columns=X.select_dtypes(exclude=['Float64','Int64']).columns
            numerical_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy='median')),
                       ("Scalar",StandardScaler())
                    
                    ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
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
            input_feture_train_pre=preprocessor.fit_transform(training_input_feature)
            input_feture_test_pre=preprocessor.transform(test_input_feature)
            train_arr=np.c_[input_feture_train_pre,np.array(traing_target)]
            test_arr=np.c_[input_feture_test_pre,np.array(test_target)]
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
