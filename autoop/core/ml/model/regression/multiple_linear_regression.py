from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    A class to represent a Multiple Linear Regression model.
    """
    def __init__(self):
        """
        Initializes the wrapper with a
        LinearRegression model from scikit-learn.
        """
        self._model = LinearRegression()

    def fit(self, X, y):
        """Fits the model to the training data.

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
            Predictions for the input data.
        """
        return self._model.predict(X)
