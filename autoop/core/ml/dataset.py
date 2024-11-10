from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class to represent a dataset artifact.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the Dataset object.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       name: str,
                       asset_path: str,
                       version: str = "1.0.0") -> Artifact:
        """
        Creates a Dataset object from a pandas DataFrame.
        Args:
            data (pd.DataFrame): The pandas DataFrame containing the data.
            name (str): The name of the dataset.
            asset_path (str): The path to the dataset asset.
            version (str): The version of the dataset.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset from the artifact.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the dataset to the artifact.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
