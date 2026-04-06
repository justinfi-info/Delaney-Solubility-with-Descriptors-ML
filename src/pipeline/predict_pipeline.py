import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading model and preprocessor")

            model_path = os.path.abspath(os.path.join('artifacts', 'model.pkl'))
            preprocessor_path = os.path.abspath(os.path.join('artifacts', 'preprocessor.pkl'))

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Ensure input is DataFrame
            if not isinstance(features, pd.DataFrame):
                raise CustomException("Input must be a pandas DataFrame", sys)

            logging.info("Applying preprocessing")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        MolLogP: float,
        MolWt: float,
        NumRotatableBonds: int,
        AromaticProportion: float
    ):
        self.MolLogP = MolLogP
        self.MolWt = MolWt
        self.NumRotatableBonds = NumRotatableBonds
        self.AromaticProportion = AromaticProportion

    def to_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("Converting custom data to DataFrame")

            custom_data_input_dict = {
                "MolLogP": [self.MolLogP],
                "MolWt": [self.MolWt],
                "NumRotatableBonds": [self.NumRotatableBonds],
                "AromaticProportion": [self.AromaticProportion],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return self.to_dataframe()