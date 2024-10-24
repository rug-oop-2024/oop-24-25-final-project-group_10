from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso


class LassoRegression(Model):
    def __init__(self, alpha=1.0):
        """
        Initializes the Lasso regression model with a regularization parameter.

        Args:
            alpha (float): Regularization strength. Must be a positive float.
            Larger values specify stronger regularization.
        """
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        """Fits the Lasso regression model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self.model.predict(X)
