import os 
import sys 

# Add the project root to the system path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from sources.exception import CustomException
from sources.logger import logging

# Your other import statements...

from sources.components.data_transformation import DataTransformationn
from sources.components.data_transformation import DataTransformationConfigg


from sources.components.model_trainer import ModelTrainerConfig
from sources.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path:str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\heart_disease.csv')

            print("Original Columns:", df.columns)

            # Replace spaces with underscores in column names
            df.columns = [col.replace(' ', '_') for col in df.columns]

            print("Modified Columns:", df.columns)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train test split initiated')

            train_set, test_set = train_test_split(df, test_size = 0.2, random_state= 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Ingestion of the data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
     obj = DataIngestion()
     train_data, test_data = obj.initiate_data_ingestion()
     print(train_data)
     data_trf_obj = DataTransformationn()
     print('la hai la')
     data_trf_obj.data_transformation_object()
     train_arr, test_arr, _ = data_trf_obj.initiate_data_transform(train_path=train_data, test_path= test_data)
     modeltrainer = ModelTrainer()
     print(modeltrainer.initiate_model_trainer(train_array= train_arr, test_array= test_arr))




