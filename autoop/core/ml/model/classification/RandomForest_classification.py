from autoop.core.ml.model.model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassModel(Model):
    """
    Random Forest Classifier model
    """
    def __init__(self):
        """
        Initialize the Random Forest Classifier model.
        """
        super().__init__(model_type='RandomForestClassModel')
        self._model = RandomForestClassifier()

    def fit(self, X, y):
        """
        Fits the Random Forest Classifier model to the training data.
        """
        self._model.fit(X, y)
        self._parameters = self._model.get_params()
        self._is_fitted = True

    def predict(self, X):
        """
        Predicts the target values for given features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._model.predict(X)
