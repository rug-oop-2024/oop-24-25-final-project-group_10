from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import base64


class Artifact(BaseModel):
    """
    A class to represent an artifact.
    """
    name: str = Field(..., title="Name of the artifact")
    asset_path: str = Field(..., title="Path to the asset")
    version: str = Field(..., title="Version of the asset")
    data: Optional[bytes] = Field(None, title="Binary data of the asset")
    metadata: Dict[str, Any] = Field(
        {}, title="Additional metadata related to the asset"
    )
    type: str = Field(
        ...,
        title="Type of the asset, e.g., 'model:torch', 'preprocessor:scaler'"
    )
    tags: List[str] = Field([], title="Tags for categorization")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def id(self) -> str:
        """
        Generate a unique ID for the artifact based on asset_path and version.
        Append `.json` to ensure compatibility with expected file format.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}_{self.version}.json"

    def __str__(self) -> str:
        """String representation of the artifact."""
        return (
            f"Artifact(name={self.name}, asset_path={self.asset_path}, "
            f"version={self.version}, "
            f"type={self.type})"
        )

    def read(self) -> bytes:
        """Read the binary data of the artifact."""
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data

    def save(self, data: bytes) -> bytes:
        """Save the binary data of the artifact."""
        self.data = data
        return data
