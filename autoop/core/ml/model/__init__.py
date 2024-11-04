from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import LassoRegression
from autoop.core.ml.model.classification import KNN
from autoop.core.ml.model.classification import LogisticModel
from autoop.core.ml.model.classification import RandomForestClassModel


REGRESSION_MODELS = [
    "multiple_linear_regression",
    "ridge_regression",
    "lasso_regression"
]
# add your models as str here

CLASSIFICATION_MODELS = [
    "KNN",
    "LogisticModel",
    "RandomForestClassModel"
]
# add your models as str here


def get_model(model_name: str) -> Model:
    """
    Returns a model object based on the name.
    Args:
        model_name (str): The name of the model.
    """
    if model_name in REGRESSION_MODELS:
        if model_name == "multiple_linear_regression":
            return MultipleLinearRegression()

        if model_name == "ridge_regression":
            return RidgeRegression()

        if model_name == "lasso_regression":
            return LassoRegression()
    if model_name in CLASSIFICATION_MODELS:
        if model_name == "KNN":
            return KNN()

        if model_name == "LogisticModel":
            return LogisticModel()

        if model_name == "RandomForestClassModel":
            return RandomForestClassModel()
