import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured during saving the object to file")
        raise CustomException(e, sys)

def evaluate_model(X_train, Y_train, X_test, Y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            #Train the model
            model.fit(X_train, Y_train)

            #Predict the model
            # Y_train_prediction = model.predict(X_train)
            Y_test_prediction = model.predict(X_test)
            print(len(Y_test_prediction))
            print(len(Y_test),len(X_test))
                  
            #Get R2 score for train and test data
            # train_model_score = r2_score(Y_train,Y_train_prediction)
            test_model_score = r2_score(Y_test, Y_test_prediction)

            report[list(models.keys())[i]] = test_model_score

            return report
        
    except Exception as e:
        logging.info("Exception occured during model training and evaluation")                                                                                               
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occured during loading the object from file")
        raise CustomException(e, sys)