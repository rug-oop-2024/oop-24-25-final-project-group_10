from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
# implemented


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects whether features in a dataset are categorical or numerical.

    Args:
    dataset (Dataset): A Dataset object containing tabular data.

    Returns:
    List[Feature]:
    A list of Feature objects with their name and detected type.
    """
    feature_list = []

    data = dataset.read()

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, feature_type=feature_type)
        feature_list.append(feature)

    return feature_list
