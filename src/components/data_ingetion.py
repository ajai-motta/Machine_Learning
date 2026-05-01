import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingection(self):
        logging.info("Enterd the data injection componet or method")
        try:
           df= pd.read_csv('notebooks\data\student_data.csv')
           logging.info('The dataset has been read')
           os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
           df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
           logging.info("Train test split initiating")
           train_set,test_set=train_test_split(df,test_size=.25,random_state=42)
           train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
           test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
           return (
               self.ingestion_config.raw_data_path,
               self.ingestion_config.test_data_path,
               self.ingestion_config.train_data_path
           )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obje=DataIngestion()
    obje.initiate_data_ingection()
    logging.info("data injection finished")