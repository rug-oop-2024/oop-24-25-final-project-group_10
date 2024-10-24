from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "accuracy",
    "precision",
    "recall"
]
# add the names (in strings) of the metrics you implement


def get_metric(name: str):
    """Returns a metric object based on the name.

    Args:
    name (str): The name of the metric.

    Returns:
    Any: An object of the metric class.
    """
    # regression metrics
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    # classification metrics
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    else:
        raise ValueError(f"Unknown metric: {name}")


class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """Calculates the metric value based on ground truth and prediction.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated metric value.
        """
        pass

# # # Regression Metrics # # #


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Mean Squared Error (MSE) metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated MSE value.
        """
        return np.mean((ground_truth - prediction) ** 2)


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error (MAE) metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Mean Absolute Error (MAE) metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated MAE value.
        """
        return np.mean(np.abs(ground_truth - prediction))


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error (RMSE) metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Root Mean Squared Error (RMSE) metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated RMSE value.
        """
        return np.sqrt(np.mean((ground_truth - prediction) ** 2))

# # # Classification Metrics # # #


class Accuracy(Metric):
    """
    Accuracy metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Accuracy metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated accuracy value.
        """
        correct = np.sum(ground_truth == prediction)
        total = len(ground_truth)
        return correct / total


class Precision(Metric):
    """
    Precision metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Precision metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated precision value.
        """
        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_positive = np.sum((ground_truth == 0) & (prediction == 1))
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    """
    Recall metric.
    """

    def __call__(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray
                 ) -> float:
        """
        Calculates the Recall metric.

        Args:
            ground_truth (np.ndarray): The actual values (ground truth).
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The calculated recall value.
        """
        true_positive = np.sum((ground_truth == 1) & (prediction == 1))
        false_negative = np.sum((ground_truth == 1) & (prediction == 0))
        return true_positive / (true_positive + false_negative)
