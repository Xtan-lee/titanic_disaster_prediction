import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.exception import CustomException

from src.utils import save_object

# @dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "titanic_disaster_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def custom_preprocessor(self, df):
        try:
            # Feature Generation

            # Get titles
            df["Title"] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

            # Mapping French Madame and Madammoiselle to Mrs and Miss respectively
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
            # Mapping Don and Dona to Sir and Madam respectively 
            df['Title'] = df['Title'].replace('Dona', 'Lady')
            df['Title'] = df['Title'].replace('Don', 'Sir')

            # Family Size
            df["Fsize"] = df["SibSp"] + df["Parch"]

            # Ticket first letters
            df["Ticket"] = df["Ticket"].apply(lambda x: str(x)[0])

            # Cabin first letters
            df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)

            return df
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_model_trainer(self,train_data_file_path, feature_transformers):
        try:
            classifiers = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
                "KNN": KNeighborsClassifier(),
                "LogReg": LogisticRegression(),
                "SVM": SVC()
            }
        

            classifier_params = {
                "Random Forest": {
                    'clf__max_depth': np.arange(3,7,1),
                    'clf__n_estimators': np.arange(50,250,100)
                },
                "XGBoost": {
                    'clf__learning_rate': np.arange(0.05, 1, 0.05),
                    'clf__max_depth': np.arange(3,7,1),
                    'clf__n_estimators': np.arange(50,250,100)
                },
                "KNN": {
                    'clf__n_neighbors': [3, 5, 7, 9],
                    'clf__weights': ['uniform', 'distance'],
                    'clf__algorithm': ['auto', 'brute'],
                },
                "LogReg": {
                    'clf__penalty': ['l1', 'l2'],
                    'clf__C': [0.001, 0.01, 0.1, 1, 10],
                    'clf__solver': ['liblinear', 'saga'],
                },
                "SVM": {
                    'clf__C': [0.001, 0.01, 0.1, 1, 10],
                    'clf__kernel': ['linear', 'rbf', 'poly'],
                    'clf__gamma': ['scale', 'auto'],
                }

            }

            model_report = {}
            df = pd.read_csv(train_data_file_path)

            for key in classifiers:
                # Combine the preprocessor and model into a single pipeline
                print(f" Starting Preprocessing, Feature Engineering and Model Initialization for {key} Classifier")
                pipeline = Pipeline(steps=[
                    ('custom_preprocessing', FunctionTransformer(func=self.custom_preprocessor)),
                    ('preprocessor', feature_transformers),
                    ('clf', classifiers[key])
                ])

                print(f"Starting Hyper-parameter selection and Model Training with Cross-Validation for {key}")
                model = GridSearchCV(estimator=pipeline, param_grid=classifier_params[key], scoring='accuracy', cv=5, verbose=False)
                model.fit(df.drop(['Survived'], axis=1), df['Survived'])
                print(f"Training completed for {key}. Cross-Validation Acurracy for best Model is {model.best_score_}.")
                
                print(f"Saving best model artifacts for {key}.")
                model_report_list = []
                model_report_list.append(model.best_score_) 
                model_report_list.append(model.best_params_)
                model_report_list.append(model.best_estimator_)

                model_report[key] = model_report_list 
                print(f"Save successful for {key}.")

            print(f"Collating results for Models in a Dataframe")
            model_report_df = pd.DataFrame.from_dict(model_report, orient='index',
                            columns=['best_score_cv', 'best_params', 'best_estimator'])
            model_report_df = model_report_df.sort_values("best_score_cv", ascending=False)
            print(model_report_df.drop(['best_estimator'], axis=1))
            best_model = model_report_df["best_estimator"].values[0]
            print(f"The best model is {model_report_df.index.values[0]} with hyper-parameters {model_report_df.best_params.values[0]}")

            print(f"Saving Collated Model Reports to CSV")
            os.makedirs('output',exist_ok=True)
            model_report_df.drop(['best_estimator'], axis=1).to_csv('output/model_report.csv')

            print(f"Saving best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f"Training Complete")
            return best_model
            
        except Exception as e:
            raise CustomException(e,sys)