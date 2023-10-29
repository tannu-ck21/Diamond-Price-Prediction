import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainingConfig:
    trained_model_ob_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_training_Config = ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Model Training Started")
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, Y_train, X_test, Y_test = (
                                                train_array[:,:-1],             # Independent Variables - X_train
                                                train_array[:,-1],              # Dependent Variable - Y_train
                                                test_array[:,:-1],              # Independent Variables - X_test
                                                test_array[:,-1]               # Dependent Variable - Y_test
            )

            # logging.info("Model Training Started")
            models = {
                        "LinearRegression" : LinearRegression(),
                        "Ridge" : Ridge(),
                        "Lasso" : Lasso(),
                        "ElasticNet" : ElasticNet(),
                        "DecisionTree" : DecisionTreeRegressor()
            }

            model_report: dict = evaluate_model(X_train,Y_train,X_test,Y_test,models)
            print(model_report)
            print("\n====================================================================================================================\n")
            logging.info(f'Model Report: {model_report}')

            #To get best model score and their name from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]              #pickel file

            print(f'Best Model: {best_model}\n R2 Score: {best_model_score}')
            print("\n====================================================================================================================\n")
            logging.info(f'Best Model: {best_model}\n R2 Score: {best_model_score}')

            logging.info("Model Training Completed")

            save_object(
                file_path = self.model_training_Config.trained_model_ob_path, 
                obj = best_model
            )

            logging.info(f"Trained Model Object Saved at {self.model_training_Config.trained_model_ob_path}")
            logging.info("Model Training Completed")
            

        except Exception as e:
            logging.info("Exception occured in model training configuration")
            raise CustomException(e,sys)
    