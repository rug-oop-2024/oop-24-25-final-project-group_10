from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import Model


class RidgeRegression(Model):
    """
    A class to represent a Ridge regression model.
    """
    def __init__(self, alpha=1.0):
        """
        Initializes the Ridge regression model with a regularization parameter.

        Args:
            alpha (float): Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
        """
        super().__init__(model_type='regression')
        self._model = Ridge(alpha=alpha)

    def fit(self, X, y):
        """Fits the Ridge regression model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        print("fit is called mf")
        self._model.fit(X, y)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_
        self._is_fitted = True
        print(self._parameters["weights"])
        print(self._parameters["weights"].tobytes())

    def predict(self, X):
        """Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
