
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import LassoRegression


REGRESSION_MODELS = [
    "multiple_linear_regression",
    "ridge_regression",
    "lasso_regression"
]
# add your models as str here

CLASSIFICATION_MODELS = [
]
# add your models as str here


def get_model(model_name: str) -> Model:
    if model_name in REGRESSION_MODELS:
        if model_name == "multiple_linear_regression":
            return MultipleLinearRegression()

        if model_name == "ridge_regression":
            return RidgeRegression()

        if model_name == "lasso_regression":
            return LassoRegression()
