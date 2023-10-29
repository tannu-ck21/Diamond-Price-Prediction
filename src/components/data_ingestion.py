import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass     #data ingestion class create

#initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

#creating a data ingestion class
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion method started")

        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset read as Pandas dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info("Train Test split started")
            train_set,test_set =train_test_split(df,test_size=0.30,random_state = 42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index = False,header = True)

            logging.info("Train Test split completed")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured in data ingestion configuration")

if __name__ == "__main__":
    logging.info("Data ingestion started")
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed")
