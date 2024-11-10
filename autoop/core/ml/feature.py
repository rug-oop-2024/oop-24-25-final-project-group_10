
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Feature class to represent a feature in a dataset.
    """
    name: str = Field(..., title="Name of the feature")
    type: Literal["numerical", "categorical"] = Field(
        ..., title="Type of the feature"
    )

    def extract(self, dataset: Dataset) -> np.ndarray:
        """Extracts the feature from the dataset.

        Args:
        dataset (Dataset): A Dataset object containing tabular data.

        Returns:
        np.ndarray: A numpy array containing the feature values.
        """
        data = dataset.read()
        return data[self.name].values

    def __str__(self) -> str:
        """
        String representation of the Feature, using name and feature type.
        """
        return f"{self.name} ({self.feature_type})"
