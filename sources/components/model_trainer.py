import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sources.exception import CustomException
from sources.logger import logging

from sources.utils import save_object
from sources.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer (self, train_array, test_array):
        try:
            logging.info('splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, : -1],
                train_array[:, -1],
                test_array[:, : -1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNeighbors Classifier": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Naive Bayes": GaussianNB(),
                "Support Vector Machine": SVC(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "KNeighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                "Naive Bayes": {},
                "Support Vector Machine": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            }
            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)
            # to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
          


            # to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            
            
            logging.info("best performing model found on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            ) # dumps the best model into pkl file
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score( y_test, predicted)

            return accuracy
        
        except Exception as e: 
            raise CustomException(e, sys)