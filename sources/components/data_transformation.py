import os 
import sys 
import numpy as np
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
                    ('scaler', StandardScaler())
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
        

    



