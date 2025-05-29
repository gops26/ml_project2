import pandas as pd, numpy as np
from src.utils import load_object
import sys, os
from src.exceptions import CustomException

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features ):
        try:
            preprocesser_filepath=os.path.join("artifacts","preprocessor.pkl")
            model_filepath=os.path.join("artifacts","model.pkl")
            model = load_object(model_filepath)
            pred = model.predict(features) 
            return pred
        except Exception as e:
            raise CustomException(e, sys)  

class CustomData:
    def __init__(
        self,
        engine_size,
        cylinders,
        fuel_type,
        fuel_consumption,      
        ):
        self.engine_size = engine_size
        self.cylinders = cylinders
        self.fuel_type= fuel_type
        self.fuel_consumption = fuel_consumption

    def get_data_as_dataframe(self):
        try:
            custom_data_dict={"engine_size":[self.engine_size],
            "cylinders":[self.cylinders],
            "fuel_type":[self.fuel_type],
            "fuel_consumption":[self.fuel_consumption],}

            df = pd.DataFrame(custom_data_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)