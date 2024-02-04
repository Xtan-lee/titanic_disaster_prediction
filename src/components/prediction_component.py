import sys

import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self):
        try:
            model_path = os.path.join("artifacts","titanic_disaster_model.pkl")
            test_data_path = os.path.join("artifacts","test.csv")
            predictions_output_path = os.path.join("output","submission.csv")
            print("Loading Model")
            model=load_object(file_path=model_path)
            print("Starting Prediction")
            test_data = pd.read_csv(test_data_path)
            predictions=model.predict(test_data)
            print("Prediction Complete. Saving Results to CSV")
            output =pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived' : predictions})
            output.to_csv('output/submission.csv', index = False)
            return (f"Prediction Task Complete. See Results in {predictions_output_path}")
        
        except Exception as e:
            raise CustomException(e,sys)