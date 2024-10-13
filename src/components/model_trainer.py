import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from  dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and Test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models={
                "LinearRegression":LinearRegression(),
                "Decision tree":DecisionTreeRegressor(),
                "KNN regressor":KNeighborsRegressor(),
                "Adaboost regressor":AdaBoostRegressor(),
                "Randomforestregressor":RandomForestRegressor()
            }

            params={
                "Decision tree":{
                    'criterion':['squared_error','absolute_error','poisson'],
                    'splitter':['best','random'],

                },
                "Randomforestregressor":{
                    'criterion':['squared_error','absolute_error','poisson'],
                    'n_estimators':[8,16,32,64,128]
                },

                "LinearRegression":{
                    'fit_intercept':[True,False]
                
                },
                "KNN regressor":{
                    'n_neighbors':[5,7,9,11]
                },
                "Adaboost regressor":{
                    'learning_rate':[0.1,0.01,0.05],
                    'n_estimators':[8,16,32]
                }
                }
            

            model_report:dict=evaluate_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,model_=models,param=params)


            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging .info("Best model found in both training and testing")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_score_value=r2_score(y_test,predicted)
            return r2_score_value

        
        except Exception as e:
            raise CustomException(e,sys)