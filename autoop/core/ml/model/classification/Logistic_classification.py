from autoop.core.ml.model.model import Model
from sklearn.linear_model import LogisticRegression


class LogisticModel(Model):
    """
    Logistic Regression model
    """
    def __init__(self):
        """
        Initialize the Logistic Regression model.
        """
        super().__init__(model_type='classification')
        self._model = LogisticRegression()

    def fit(self, X, y):
        """
        Fits the Logistic Regression model to the training data.
        """
        print("fit is called mf")
        self._model.fit(X, y)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_
        self._is_fitted = True
        print(self._parameters["weights"])
        print(self._parameters["weights"].tobytes())

    def predict(self, X):
        """
        Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
