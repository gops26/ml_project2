from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import *
from dataclasses import dataclass
import os, sys

@dataclass
class ModelTrainerConfig:
    trained_model_save_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, trainarr, testarr):
        logging.info("model trainer initiated")
        try:
            X_train, y_train = trainarr[:, :-1], trainarr[:, -1]
            X_test, y_test = testarr[:, :-1], testarr[:, -1]
        
            models = {
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor()
            }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "Ridge":{},
                "Lasso":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "KNeighborsRegressor":{
                    "n_neighbors":[2,3,4,5,6],
                    "algorithm":["ball_tree", "kd_tree", "brute"],
                    "weights":["uniform", "distance"],
                    "leaf_size":[10,20,30,40,50]
                }
            }

            model_report = evaluate_model(X_train, X_test, y_train, y_test, models, params )
            best_model_score=max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(best_model)

            if best_model_score < 0.6:
                raise CustomException("no best model found")
            logging.info("model training finished with best model and fitted sucessful")
            
            save_object(filepath=self.config.trained_model_save_path, obj=best_model)
            
            return best_model_score
        except Exception as e:
            raise CustomException(e,sys) 
