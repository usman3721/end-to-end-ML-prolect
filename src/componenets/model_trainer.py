import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomExecption
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass
import os
import sys
import warnings



@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifact',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
                )
            
            
            models={
                "linear regression": LinearRegression(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False),
                "AdaBoostRegressor":AdaBoostRegressor(),
            }
            
            
         
            model_report:dict=evaluate_models(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomExecption("No best model found")
            
            
            logging.info("Best model found on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test,predicted)
            return r2_square       
        
        
         
        except Exception as e:
            raise CustomExecption(e,sys)