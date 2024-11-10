from abc import ABC, abstractmethod
import numpy as np

REGRESSION_METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error"
]

CLASSIFICATION_METRICS = [
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
    if name in REGRESSION_METRICS:
        if name == "mean_squared_error":
            return MeanSquaredError()

        if name == "mean_absolute_error":
            return MeanAbsoluteError()

        if name == "root_mean_squared_error":
            return RootMeanSquaredError()

    if name in CLASSIFICATION_METRICS:
        if name == "accuracy":
            return Accuracy()

        if name == "precision":
            return Precision()

        if name == "recall":
            return Recall()


class Metric(ABC):
    """
    Base class for all metrics.
    """
    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Evaluates the metric based on ground truth and prediction.
        """
        return self(ground_truth, prediction)

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
        class_labels = np.unique(ground_truth)
        precisions = []
        for class_label in class_labels:
            true_positive = np.sum((ground_truth == class_label) &
                                   (prediction == class_label))
            false_positive = np.sum((ground_truth != class_label) &
                                    (prediction == class_label))

            denominator = true_positive + false_positive
            if denominator == 0:
                precisions.append(0.0)
            else:
                precisions.append(true_positive / denominator)

        macro_precision = np.mean(precisions)
        return macro_precision


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
        class_labels = np.unique(ground_truth)
        recalls = []
        for class_label in class_labels:
            true_positive = np.sum((ground_truth == class_label) &
                                   (prediction == class_label))
            false_negative = np.sum((ground_truth == class_label) &
                                    (prediction != class_label))
            denominator = true_positive + false_negative
            if denominator == 0:
                recalls.append(0.0)
            else:
                recalls.append(true_positive / denominator)
        macro_recall = np.mean(recalls)
        return macro_recall
