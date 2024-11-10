from autoop.core.ml.model.model import Model
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticModel(Model):
    """
    Logistic Regression model
    """
    def __init__(self) -> None:
        """
        Initialize the Logistic Regression model.
        """
        super().__init__(model_type='LogisticModel')
        self._model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Logistic Regression model to the training data.
        """
        self._model.fit(X, y)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
