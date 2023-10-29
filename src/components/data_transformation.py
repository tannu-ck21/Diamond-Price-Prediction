from sklearn.impute import SimpleImputer  #Handling Missing Values
from sklearn.preprocessing import StandardScaler  #Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder  #Handling Categorical Data
from sklearn.pipeline import Pipeline  #Pipeline
from sklearn.compose import ColumnTransformer #Pipeline

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "processor.pkl")   # Preprocessor Object pickle File Path

#Data Transformation Configuration Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):               #pickle file -> feature engineering

        try:
            logging.info("Data Transformation Configuration Started")
            #Define which column should be ordial encoding and which column should be scaled
            categorical_cols =["cut","color","clarity"]
            numerical_cols = ["carat","depth","table","x","y","z"]

            #Define the custom ranking for each ordinal column :   0/0/0-1.000 carat means cut/color/clarity grade - carat weighs
            cut_ranking = ["Fair","Good", "Very Good", "Premium", "Ideal"]
            color_ranking = ["J","I","H","G","F","E","D"]
            clarity_ranking = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            #Define the pipeline
            logging.info("Pipeline Started")

            #Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("ordinal", OrdinalEncoder(categories=[cut_ranking,color_ranking,clarity_ranking])),
                    ('scaler', StandardScaler())
                ]
            )

            #Full Pipeline : combininng both pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            logging.info("Pipeline Completed")
            logging.info("Data Transformation Configuration Completed")

            return preprocessor #Full Pipeline
        
        except Exception as e:
            logging.info("Exception occured in data transformation configuration")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Data Transformation Started")

            # Read the data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info("Reading Test and Train Data Completed")
            logging.info("Train Dataframe Head : \n{}".format(train_data.head().to_string()))
            logging.info("Test Dataframe Head : \n{}".format(test_data.head().to_string()))
            
            # Preprocessor object
            preprocessing_obj = self.get_data_transformation_object()            #pickle file
            logging.info("Preprocessor Object Created")

            target_cols = "price"
            drop_cols = [target_cols, "id"]

            # Separate the independent and dependent variable
            input_feature_train_data = train_data.drop(drop_cols, axis=1)   #independent variable
            target_feature_train_data = train_data[target_cols]             #dependent variable

            input_feature_test_data = test_data.drop(drop_cols, axis=1)     #independent variable
            target_feature_test_data = test_data[target_cols]               #dependent variable

            logging.info("Independent and Dependent Variable Separated")

            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            logging.info("Applying processing object on train and test dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            # Save the preprocessor object -> utils.py
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor pickle file is Created and Saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            logging.info("Exception occured in data transformation")
            raise CustomException(e,sys)











#             logging.info("Data Transformation Started")

#             # Read the data
#             train_data = pd.read_csv(train_data_path)
#             test_data = pd.read_csv(test_data_path)

#             # Separate the target variable
#             train_data = train_data.drop("price", axis=1)
#             test_data = test_data.drop("price", axis=1)

#             # Transform the data
#             train_data = preprocessor.fit_transform(train_data)
#             test_data = preprocessor.transform(test_data)

#             # Save the preprocessor object
#             os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_ob_file_path), exist_ok=True)
#             joblib.dump(preprocessor, self.data_transformation_config.preprocessor_ob_file_path)

#             # Save the transformed data
#             train_data.to_csv(os.path.join("artifacts", "train_data.csv"), index=False)
#             test_data.to_csv(os.path.join("artifacts", "test_data.csv"), index=False)

#             logging.info("Data Transformation Completed")
