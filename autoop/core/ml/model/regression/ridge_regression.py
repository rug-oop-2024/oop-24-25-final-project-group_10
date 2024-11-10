from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import Model
import numpy as np


class RidgeRegression(Model):
    """
    A class to represent a Ridge regression model.
    """
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the Ridge regression model with a regularization parameter.

        Args:
            alpha (float): Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
        """
        super().__init__(model_type='ridge_regression')
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Ridge regression model to the training data.

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
