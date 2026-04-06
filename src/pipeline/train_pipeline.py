import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ...existing code...
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main() -> None:
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )

    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Training completed. R2: {r2}")


if __name__ == "__main__":
    main()