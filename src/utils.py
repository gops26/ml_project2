import dill,pickle
import os,sys
from src.logger import logging
from src.exceptions import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sys
import numpy as np


def save_object(filepath:str, obj:any):
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj,file)
        logging.info(f"saved object{obj.__str__} to filepath {filepath} ") 
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(filepath):
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, models:dict, param:dict):
    try:
        report = {}
        for i in range(len(list(models.values()))):
            model_name=list(models.keys())[i]
            logging.info(f"model evaluation begins {model_name}")
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"grid search cv begins for {model_name}")
            
            gs = GridSearchCV(model, param_grid=para)
            gs.fit(X_train, y_train)
            
            logging.info(f"grid search cv ended for {model_name}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            logging.info(f"model evaluation finished for  {model_name}")

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)





