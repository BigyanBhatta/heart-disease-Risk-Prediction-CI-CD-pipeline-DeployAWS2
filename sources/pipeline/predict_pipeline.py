import sys
import pandas as pd
from sources.exception import CustomException
from sources.utils import load_object

class PredictPipeline:
    def __init__ (self):
        pass


    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            print('before loading')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            print('after processing')
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Gender: str, education: str, prevalentStroke: str, age: int, cigsPerDay: int,
                 sysBP: float, diaBP: float, BMI: float, heartRate: int, glucose: int,
                 currentSmoker: int, BPMeds: int, prevalentHyp: int, diabetes: int, totChol: int
                 ):
        self.Gender = Gender
        self.education = education
        self.prevalentStroke = prevalentStroke
        self.age = age
        self.cigsPerDay = cigsPerDay
        self.sysBP = sysBP
        self.diaBP = diaBP
        self.BMI = BMI
        self.heartRate = heartRate
        self.glucose = glucose
        self.currentSmoker = currentSmoker
        self.BPMeds = BPMeds
        self.prevalentHyp = prevalentHyp
        self.diabetes = diabetes
        self.totChol = totChol

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "education": [self.education],
                "prevalentStroke": [self.prevalentStroke],
                "age": [self.age],
                "cigsPerDay": [self.cigsPerDay],
                "sysBP": [self.sysBP],
                "diaBP": [self.diaBP],
                "BMI": [self.BMI],
                "heartRate": [self.heartRate],
                "glucose": [self.glucose],
                "currentSmoker": [self.currentSmoker],
                "BPMeds": [self.BPMeds],
                "prevalentHyp": [self.prevalentHyp],
                "diabetes": [self.diabetes],
                "totChol": [self.totChol]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)