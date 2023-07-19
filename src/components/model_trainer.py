import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sys

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from src.utils import save_object,evaluate_model
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                        "Linear Regression":LinearRegression(),
                        "Lasso":Lasso(),
                        "Ridge":Ridge(),
                        "K Neighbors Regressor":KNeighborsRegressor(),
                        "Decision Tree":DecisionTreeRegressor(),
                        "Random Forest":RandomForestRegressor(),
                        "XGBRegressor":XGBRegressor(),
                        "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor":AdaBoostRegressor()
                    }
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models=models)
            print(model_report)
            best_model_score=max(sorted(model_report.values()))
            print(best_model_score)
            print(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print(best_model_name)
            best_model=models[best_model_name]
            if best_model_score<.6:
                raise CustomException("No best model")
            logging.info("Found the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_Score=r2_score(y_test,predicted)
            return r2_Score

        except Exception as e:
            raise CustomException(e,sys)



