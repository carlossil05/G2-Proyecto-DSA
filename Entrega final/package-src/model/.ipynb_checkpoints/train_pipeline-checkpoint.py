import numpy as np
import pandas as pd

from .config.core import config
from .pipeline import housing_pipe
from .processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # Load pre-split training dataset
    data = load_dataset(file_name=config.app_config.train_data_file)

    # Separate target and features
    X_train = data[config.model.features]
    y_train = data[config.model.target]

    # Fit the regression model
    housing_pipe.fit(X_train, y_train)

    # Persist trained pipeline
    save_pipeline(pipeline_to_persist=housing_pipe)


if __name__ == "__main__":
    run_training()
