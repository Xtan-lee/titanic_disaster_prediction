import sys
import os

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.prediction_component import PredictPipeline

from src.utils import save_object



class DataTransformation:
    def __init__(self):
        pass
        # self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # Final segragation of features because some integer columns are categorical columns
            num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Fsize']
            cat_features = ['Sex', 'Ticket', 'Embarked','Title']

            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            # Combine the preprocessing steps for numerical and categorical features
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ],remainder='drop')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":

    os.chdir('..')

    print("Checking if there is an existing saved model")
    trained_model_file_path = os.path.join("artifacts", "titanic_disaster_model.pkl")

    if os.path.exists(trained_model_file_path):
        print("Using existing saved model")
        prediction_component = PredictPipeline()
        predictor = prediction_component.predict()
        print(predictor)

    else:
        print("No Pre-saved model found. Starting new process")
        data_transformation=DataTransformation()
        feature_transformers = data_transformation.get_data_transformer_object()
    
        train_data_file_path = os.path.join('artifacts', "train.csv")
        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(train_data_file_path,feature_transformers)

        prediction_component = PredictPipeline()
        predictor = prediction_component.predict()
        print(predictor)