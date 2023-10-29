import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,new_data):
        try:
            preprocessor_path = os.path.join("artifacts", "processor.pkl")
            model_path = os.path.join("artifacts", "trained_model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(new_data)

            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured during prediction")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 carat : float, 
                 depth : float,
                 table : float,
                 x : float,
                 y : float,
                 z : float,
                 cut : str,
                 color : str,
                 clarity : str
                 ):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.clarity = clarity
        self.cut = cut


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat" : [self.carat],
                "depth" : [self.depth],
                "table" : [self.table],
                "x" : [self.x],
                "y" : [self.y],
                "z" : [self.z],
                "cut" : [self.cut],
                "color" : [self.color],
                "clarity" : [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df
        
        except Exception as e :
            logging.info("Exception occured in Custom Data Class")
            raise CustomException(e,sys) 