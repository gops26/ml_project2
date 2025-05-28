from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split
import sys
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logging.info("this is ingestion page")


@dataclass
class DataIngestionConfig:
    train_save_path = os.path.join("artifacts", "train.csv")
    test_save_path = os.path.join("artifacts", "test.csv")
    raw_save_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("data ingestion starting ........")

            df = pd.read_csv("database/data.csv", index_col=0)

            os.makedirs(os.path.dirname(
                self.data_ingestion_config.train_save_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_save_path,
                      index=False, header=True)
            
            logging.info("made database cOpy...............")

            logging.info("initiating train test split")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.data_ingestion_config.train_save_path, index=False, header=True)
            test_set.to_csv(
                self.data_ingestion_config.test_save_path, index=False, header=True)

            logging.info("data ingestion sucessful")
            return (self.data_ingestion_config.train_save_path, self.data_ingestion_config.test_save_path)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    print(train_path, test_path)

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_preprocessing(train_path, test_path)
    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)
