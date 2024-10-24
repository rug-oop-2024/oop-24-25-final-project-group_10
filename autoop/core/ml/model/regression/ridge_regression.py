from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import Model


class RidgeRegression(Model):
    def __init__(self, alpha=1.0):
        """
        Initializes the Ridge regression model with a regularization parameter.

        Args:
            alpha (float): Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
        """
        self._model = Ridge(alpha=alpha)

    def fit(self, X, y):
        """Fits the Ridge regression model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        self._model.fit(X, y)

    def predict(self, X):
        """Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
