from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso
import numpy as np


class LassoRegression(Model):
    """
    A class to represent a Lasso regression model.
    """
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the Lasso regression model with a regularization parameter.

        Args:
            alpha (float): Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
        """
        super().__init__(model_type='lasso_regression')
        self._model = Lasso(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Lasso regression model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        self._model.fit(X, y)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
