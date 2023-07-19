# Read Data and split to train and test

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.components.data_transformation import DataTransformatioConfig,DataTransformation
@dataclass#decorator used to define class variables without init
class DataIngestionConfig:#any input we require will happen through this class
    train_data_path=os.path.join('artifact',"train.csv")#artifact folder will have outputs
    test_data_path=os.path.join('artifact',"test.csv")
    raw_data_path=os.path.join('artifact',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()#all tree paths will be saves inside this variable
    
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or compoonent")
        try:
            df=pd.read_csv('notebook/data/stud.csv')#change this line of code based on whether you are reading from database or api
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test set Initiated")
            train_set,test_set=train_test_split(df,test_size=.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ =="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))