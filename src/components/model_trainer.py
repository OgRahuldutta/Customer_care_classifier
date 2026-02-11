import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            models={
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "Xgboost Classifier": XGBClassifier()
            }

            params = {
                "Logistic Regression":{
                    'C': [0.1, 1.0, 10],
                    'solver': ['liblinear']
                },  
                "Random Forest":{
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                },
                "Gradient Boosting":{
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "AdaBoost Classifier":{
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                "Decision Tree":{
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "K-Neighbors Classifier":{
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "CatBoost Classifier":{
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5]
                },
                "Xgboost Classifier":{
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }

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

            logging.info(f"Best model: {best_model_name} | ROC-AUC: {best_model_score}")

            if best_model_score < 0.70:
                raise Exception("No model achieved minimum ROC-AUC threshold (0.70)")

            # ✅ Save trained best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluation metrics
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                final_score = roc_auc_score(y_test, y_proba)
            else:
                y_pred = best_model.predict(X_test)
                final_score = roc_auc_score(y_test, y_pred)

            logging.info(f"Final ROC-AUC on test set: {final_score}")

            y_pred_labels = best_model.predict(X_test)
            logging.info("\n" + classification_report(y_test, y_pred_labels))

            return best_model_name, final_score

        except Exception as e:
            raise CustomException(e, sys)