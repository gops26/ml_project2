from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exceptions import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

    def get_preprocessing_obj(self):
        try:
            num_features = ['engine_size', 'cylinders',
                            'fuel_consumption', 'co2_emissions']
            cat_features = ['fuel_type']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehotencoder", OneHotEncoder())
                ]
            )

            logging.info("pipeline creation completed")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("categorical pipeline", cat_pipeline, cat_features)
                ]
            )
            logging.info("preprocessing obj completed")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_preprocessing(self, train_path, test_path):
        try:
            logging.info("preprocessing starting... ")
            preprocessor = self.get_preprocessing_obj()

            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            target_feature_name = "co2_emissions"

            input_feature_train = train_df.drop(target_feature_name)
            target_feature_train = train_df[target_feature_name]

            input_feature_test = test_df.drop(target_feature_name)
            target_feature_test = test_df[target_feature_name]

            logging.info("applying transformation on data")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor.transform(input_feature_test)

            train_arr = np.c_[
               target_feature_train, np.array(input_feature_train_arr) 
            ]
            
            test_arr = np.c_[
               target_feature_test, np.array(input_feature_test_arr) 
            ]

            save_object(
                filepath=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return (
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
