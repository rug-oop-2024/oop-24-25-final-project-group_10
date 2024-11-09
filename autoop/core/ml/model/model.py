from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    """Abstract base class for machine learning models."""

    def __init__(self,
                 model_type: Literal['classification', 'regression'],
                 parameters: dict = None):
        """Initialize the base model.

        Args:
            model_type (Literal['classification', 'regression']):
            The type of the model.
            parameters (dict, optional):
            The model's parameters (weights, coefficients, etc.).
        """
        self._type = model_type
        self._parameters = deepcopy(parameters) if parameters else {}
        self._is_fitted = False
        self._artifact = None
        self._metadata = {"metrics": [], "dataset": None}

    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        """Fit the model to the data.

        Args:
            X (np.ndarray): The training data (features).
            y (np.ndarray): The target values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions based on the fitted model.

        Args:
            X (np.ndarray): The input data for predictions.

        Returns:
            np.ndarray: The predictions made by the model.
        """
        pass

    def save(self, model_name: str, model_version: str) -> Artifact:
        """
        Save the model as an artifact.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving.")
        self._artifact = Artifact(
            name=model_name,
            asset_path=(
                f"models/{self._type}_models/{model_name}_{model_version}.pkl"
            ),
            version=model_version,
            type="model",
            data=np.array(self._parameters).tobytes(),
            metadata=self._metadata
        )
        return self._artifact

    def load(self, artifact: Artifact):
        """Load the model from an artifact.

        Args:
            artifact (Artifact): The artifact to load the model from.
        """
        self._artifact = deepcopy(artifact)
        self._parameters = np.frombuffer(self._artifact.data,
                                         dtype=np.float64)
        self._is_fitted = True
        print(f"Model loaded from artifact at {artifact.asset_path}.")

    def __str__(self):
        """String representation of the model."""
        return f"Model(type={self._type}, is_fitted={self._is_fitted})"

    @property
    def parameters(self):
        """Returns the model's parameters (coefficients and intercept)."""
        return self._parameters

    @property
    def type(self) -> str:
        """Returns the model's type (classification or regression)."""
        return self._type

    @property
    def metadata(self) -> dict:
        """Returns the model's metadata."""
        return deepcopy(self._metadata)

    def set_metric_score(self, metric: str, score: float):
        """Set the model's metric score."""
        self._metadata["metrics"].append([metric, score]) 

    def set_trained_dataset(self, dataset: str):
        """Set the model's trained on attribute."""
        self._metadata["dataset"] = dataset
