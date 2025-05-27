from dataclasses import dataclass
import os, sys
from src.logger import logging
from src.exceptions import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

    def get_preprocessing_obj(self):
        num_features= ['engine_size', 'cylinders', 'fuel_consumption', 'co2_emissions']
        cat_features = ['fuel_type']

        num_pipeline = Pipeline(
            steps=[
                ("scaler",StandardScaler(with_mean=False))
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("onehotencoder", OneHotEncoder())
            ]
        )

        logging.info("pipeline creation completed")
        preprocessor= ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, num_features),
                ("categorical pipeline",cat_pipeline, cat_features )
            ]
        )
        logging.info("preprocessing obj completed")

        return preprocessor

    def initiate_preprocessing(self, train_path, test_path):
        pass
