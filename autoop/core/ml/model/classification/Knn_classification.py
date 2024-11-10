from autoop.core.ml.model.model import Model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN(Model):
    """
    KNN model for classification tasks.
    """
    def __init__(self, n_neighbors: int = 5) -> None:
        """
        Initializes the KNN model with the number of neighbors.

        Args:
            n_neighbors (int):
            Number of neighbors to consider for classification.
        """
        super().__init__(model_type='KNN')
        self._n_neighbors = n_neighbors
        self._model = KNeighborsClassifier(n_neighbors=self._n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the KNN model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        self._model.fit(X, y)
        self._parameters = self._model.get_params(deep=True)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted.")
        return self._model.predict(X)
