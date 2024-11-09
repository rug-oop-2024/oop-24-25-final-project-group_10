from autoop.core.ml.model.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNN(Model):
    def __init__(self, n_neighbors=5):
        """
        Initializes the KNN model with the number of neighbors.

        Args:
            n_neighbors (int):
            Number of neighbors to consider for classification.
        """
        super().__init__(model_type='classification')
        self._n_neighbors = n_neighbors
        self._model = KNeighborsClassifier(n_neighbors=self._n_neighbors)

    def fit(self, X, y):
        """
        Fits the KNN model to the training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        print("fit is called mf")
        self._model.fit(X, y)
        print(self._model.get_params())
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
