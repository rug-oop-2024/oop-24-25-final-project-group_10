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

    def save(self, path: str):
        """Save the model as an artifact.

        Args:
            path (str): Path to save the model artifact.
        """
        self._artifact = Artifact(
            asset_path=path,
            version="1.0",
            data=np.array(self._parameters["weights"]),
            metadata={"model_type": self._type},
            type=f"model:{self._type}",
            tags=["machine_learning", "model"]
        )
        print(f"Model saved as artifact at {path}.")

    def load(self, artifact: Artifact):
        """Load the model from an artifact.

        Args:
            artifact (Artifact): The artifact to load the model from.
        """
        self._artifact = deepcopy(artifact)
        self._parameters["weights"] = artifact.data
        self._is_fitted = True
        print(f"Model loaded from artifact at {artifact.asset_path}.")

    def __str__(self):
        """String representation of the model."""
        return f"Model(type={self._type}, is_fitted={self._is_fitted})"
    
    @property
    def parameters(self):
        """Returns the model's parameters (coefficients and intercept)."""
        return {
            "coefficients": self._model.coef_,
            "intercept": self._model.intercept_
        }
