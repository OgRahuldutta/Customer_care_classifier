import os
import sys
import dill
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_model = None
        best_score = 0

        for model_name, model in models.items():

            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            # 🔥 Correct ROC-AUC calculation
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            score = roc_auc_score(y_test, y_proba)

            report[model_name] = score

            if score > best_score:
                best_score = score
                best_model = model

        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)
