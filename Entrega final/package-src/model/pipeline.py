from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from model.config.core import config
from model.processing import features as pp

housing_pipe = Pipeline(
    steps=[
        (
            "gradient_boosting_model",
            GradientBoostingRegressor(
                n_estimators=config.model.n_estimators,
                learning_rate=config.model.learning_rate,
                max_depth=config.model.max_depth,
                random_state=config.model.random_state,
            ),
        ),
    ]
)