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
        super().__init__(model_type='multiple_linear_regression')
        self._model = LinearRegression()
        if self._is_fitted:
            self._model.set_params(**self._parameters)

    def fit(self, X, y):
        """Fits the model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        self._model.fit(X, y)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_
        self._is_fitted = True

    def predict(self, X):
        """Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions for the input data.
        """
        return self._model.predict(X)
