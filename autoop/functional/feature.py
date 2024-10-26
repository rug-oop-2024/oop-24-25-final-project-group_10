from typing import List
import pandas as pd
from io import BytesIO
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects whether features in a dataset are categorical or numerical.

    Args:
    dataset (Dataset): A Dataset object containing tabular data.

    Returns:
    List[Feature]:
    A list of Feature objects with their name and detected type.
    """
    data = dataset.read()
    features = []
    # check if the data is in bytes format
    if isinstance(data, bytes):
        data = pd.read_csv(BytesIO(data))
    for column in data.columns:
        if data[column].dtype == "object":
            feature_type = "categorical"
        else:
            feature_type = "numerical"
        features.append(Feature(name=column, type=feature_type))
    return features