import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            num_features = [
                "Age",
                "Tenure_Months",
                "Monthly_Charges"
            ]

            cat_features = [
                "Internet_Service",
                "Contract_Type",
                "Payment_Method",
                "Gender",
                "Tech_Support"
            ]

            # =========================
            # Numeric Pipeline
            # =========================
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # =========================
            # Categorical Pipeline
            # =========================
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False  # Important for XGBoost
                ))
            ])

            # =========================
            # Column Transformer
            # =========================
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column = "Churn"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # =========================
            # Preprocessing
            # =========================
            preprocessor = self.get_data_transformation_object()

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Convert to numpy explicitly
            X_train_processed = np.array(X_train_processed)
            X_test_processed = np.array(X_test_processed)

            # =========================
            # Encode Target (ONLY HERE)
            # =========================
            label_encoder = LabelEncoder()

            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            # =========================
            # Merge Features + Target
            # =========================
            train_arr = np.c_[X_train_processed, y_train_encoded]
            test_arr = np.c_[X_test_processed, y_test_encoded]

            # =========================
            # Save Artifacts
            # =========================
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            save_object(self.config.label_encoder_file_path, label_encoder)

            logging.info("Preprocessing and label encoding completed")

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
