from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


class RegressionWrapper(Model):
    """
    Wrapper class for regression models
    Utilizing the sklearn library
    """
    def __init__(self, model):
        """
        Initialize the regression model with the given model.
        :param model: The regression model to wrap.
        """
        self._model = model

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class LinearRegressionModel(RegressionWrapper):
    """
    Linear Regression model without regularization.
    """
    def __init__(self, fit_intercept=True, normalize=False):
        """
        Initialize the Linear Regression model with optional hyperparameters.
        :param fit_intercept
        Whether to calculate the intercept for this model.
        :param normalize:
        This parameter is ignored when fit_intercept is set to False.
        If True, the regressors X will be normalized before fitting.
        """
        super().__init__(LinearRegression(fit_intercept=fit_intercept,
                                          normalize=normalize))


class LassoModel(RegressionWrapper):
    """
    Lasso regression model
    """
    def __init__(self, alpha=0.1):
        """
        Initialize the Lasso model with the given alpha.
        :param alpha: Regularization strength; must be a positive float.
        """
        self._alpha = alpha
        super().__init__(Lasso(self._alpha))


class RidgeModel(RegressionWrapper):
    """
    Ridge regression model
    """
    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge model with the given alpha.
        :param alpha: Regularization strength; must be a positive float.
        """
        self._alpha = alpha
        super().__init__(Ridge(self._alpha))
