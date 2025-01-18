import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline.
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                                 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            categorical_columns = ['Geography', 'Gender']

            # Define numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Define categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Log column details
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Specify columns to drop
            columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']

            # Drop specified columns if they exist
            train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')
            test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], errors='ignore')

            logging.info(f"Dropped columns: {columns_to_drop}")
            logging.info(f"Train DataFrame columns after dropping: {train_df.columns}")
            logging.info(f"Test DataFrame columns after dropping: {test_df.columns}")

            # Define the target column
            target_column_name = 'Exited'

            # Check if the target column exists
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in train or test data.")

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on the data.")

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
