import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment]


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Keep training quick and reliable on Windows by avoiding large GridSearchCV
            # (which can be slow and can hit process-spawn permission issues).
            models: dict[str, object] = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }

            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor(eval_metric="rmse", random_state=42)

            if CatBoostRegressor is not None:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False, random_state=42)

            best_model_name: str | None = None
            best_model = None
            best_score = float("-inf")

            logging.info("Training candidate models")
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = r2_score(y_test, preds)
                    logging.info(f"Model {name} r2_score={score}")

                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = model
                except Exception as e:
                    logging.info(f"Skipping model {name} due to error: {e}")

            if best_model is None or best_model_name is None:
                raise CustomException("No model could be trained successfully", sys)

            if best_score < 0.6:
                raise CustomException(
                    f"No good model found with acceptable performance (best={best_model_name}, r2={best_score})",
                    sys,
                )

            logging.info(f"Best model: {best_model_name} with score: {best_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
