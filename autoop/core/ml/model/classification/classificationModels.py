from autoop.core.ml.model.model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ClassifierWrapper(Model):
    def __init__(self, model):
        self._model = model

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class KNNModel(ClassifierWrapper):
    """
    K-Nearest Neighbors model
    """
    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN model with the given number of neighbors.
        :param n_neighbors:
        Number of neighbors to use for classification.
        """
        self._n_neighbors = n_neighbors
        super().__init__(KNeighborsClassifier(self._n_neighbors))


class LogisticModel(ClassifierWrapper):
    """
    Logistic Regression model
    """
    def __init__(self):
        """
        Initialize the Logistic Regression model.
        """
        super().__init__(LogisticRegression())


class RandomForestClassModel(ClassifierWrapper):
    """
    Random Forest Classifier model
    """
    def __init__(self):
        """
        Initialize the Random Forest Classifier model.
        """
        super().__init__(RandomForestClassifier())
