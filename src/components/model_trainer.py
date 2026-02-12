import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('deployment','artifacts', 'model.pkl')
    label_encoder_file_path: str = os.path.join('deployment','artifacts', 'label_encoder.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # =========================
            # Encode target labels
            # =========================
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            save_object(
                file_path=self.model_trainer_config.label_encoder_file_path,
                obj=le
            )

            # =========================
            # Handle class imbalance (CRITICAL)
            # =========================
            neg, pos = np.bincount(y_train)
            scale_pos_weight = neg / pos

            # =========================
            # Strong Models
            # =========================
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "Xgboost Classifier": XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    learning_rate=0.03,
                    n_estimators=600,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=1,
                    reg_lambda=1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42
                )
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },

                "Random Forest": {
                    "n_estimators": [200, 300],
                    "max_depth": [None, 10, 20]
                },

                "Xgboost Classifier": {
                    "n_estimators": [400, 600],
                    "learning_rate": [0.02, 0.05],
                    "max_depth": [4, 5],
                    "subsample": [0.8, 1],
                    "colsample_bytree": [0.8, 1]
                }
            }

            # =========================
            # Train + Evaluate
            # =========================
            model_report, best_model = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"ROC-AUC: {best_model_score}")

            if best_model_score < 0.75:
                raise Exception("Model performance below 0.75 ROC-AUC")

            # Save best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # =========================
            # Final Evaluation
            # =========================
            y_proba = best_model.predict_proba(X_test)[:, 1]
            final_score = roc_auc_score(y_test, y_proba)

            logging.info(f"Final ROC-AUC on Test Set: {final_score}")

            y_pred = best_model.predict(X_test)
            logging.info("\n" + classification_report(y_test, y_pred))

            return best_model_name, final_score

        except Exception as e:
            raise CustomException(e, sys)
