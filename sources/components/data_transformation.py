import os 
import sys 
import numpy as np
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sources.utils import save_object

from dataclasses import dataclass

from sources.logger import logging
from sources.exception import CustomException

@dataclass
class DataTransformationConfigg:
    preprocessor_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformationn:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfigg()
        logging.info('object created')
    
    def data_transformation_object(self):
        try:    
            numerical_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'currentSmoker', 'BPMeds','prevalentHyp', 'diabetes' ]
            categorical_columns = ['Gender', 'education', 'prevalentStroke']
            logging.info('preprocessor creation starting')

            num_pipe = Pipeline(
                steps= [
                    ('standard_scaler', StandardScaler())
                ]
            )
            cat_pipe = Pipeline(
                steps = [
                    ('one_hot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            preprocess = ColumnTransformer(
                transformers=[
                    ('num_pipe', num_pipe, numerical_columns),
                    ('cat_pipe', cat_pipe, categorical_columns)
                ]
            )
            logging.info('preprocessor created')
            return preprocess
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(train_df.columns)
            logging.info('read train and test data')
            preprocess_obj = self.data_transformation_object()
            logging.info('preprocessor object obtained')
            target_column = 'Heart_ stroke'
            input_feat_train_df = train_df.drop(columns = [target_column])
            target_feat_train_df = train_df[target_column]

            input_feat_test_df = test_df.drop(columns = [target_column])
            target_feat_test_df = test_df[target_column]
            
            logging.info('applying preprocessor object in the input feature')
            input_feat_train_arr = preprocess_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr = preprocess_obj.fit_transform(input_feat_test_df)

            train_array = np.c_ [
                input_feat_train_arr, np.array(target_feat_train_df)
            ]
            test_array = np.c_ [input_feat_test_arr, np.array(target_feat_test_df)]

            logging.info('preprocessor completed')

            save_object(file_path= self.data_transformation_config.preprocessor_path, obj= preprocess_obj)

            logging.info('object saved')
            return train_array, test_array, self.data_transformation_config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
        

    



