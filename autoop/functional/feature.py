from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects whether features in a dataset are categorical
    or numerical based on their content.
    This function handles datasets read from CSV files with appropriate
    delimiters detected automatically.

    Args:
        dataset (Dataset): A Dataset object containing tabular data.

    Returns:
        List[Feature]: A list of Feature objects with their name
        and detected type.
    """
    data = dataset.read()

    features = []
    for column in data.columns:
        if pd.api.types.is_string_dtype(data[column]):
            unique_vals = data[column].dropna().unique()
            try:
                pd.Series(unique_vals).astype(float)
                feature_type = 'numerical'
            except ValueError:
                feature_type = 'categorical'
        elif pd.api.types.is_numeric_dtype(data[column]):
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'

        features.append(Feature(name=column, type=feature_type))
    return features
