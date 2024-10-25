from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import base64
# implemented


class Artifact(BaseModel):
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

    @property
    def id(self) -> str:
        """
        Generate a unique ID for the artifact based on asset_path and version.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def __str__(self) -> str:
        """String representation of the artifact."""
        return (
            f"Artifact(asset_path={self.asset_path}, version={self.version}, "
            f"type={self.type})"
        )

    def read(self) -> bytes:
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data