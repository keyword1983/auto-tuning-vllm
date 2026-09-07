"""Abstract base class for inference engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EngineAdapter(ABC):
    """Abstract adapter defining engine-specific parameter mapping and behaviors."""

    name: str = "base"
    default_port: int = 8000
    openai_compatible: bool = True

    @abstractmethod
    def get_default_parameter_space(self) -> Dict[str, Any]:
        """Return the default hyperparameter search space for this engine."""
        pass

    @abstractmethod
    def extract_baseline_parameters(self, serving_template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract baseline parameter values from an AFSBox ModelServing template."""
        pass

    @abstractmethod
    def map_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map trial hyperparameters to an AFSBox ModelServing spec patch.

        Distinguishes between top-level ModelServing CR fields (handled natively
        by AFSBox controller) and engine-specific CLI flags (placed in extraCommand).
        """
        pass

    @abstractmethod
    def build_server_command(
        self,
        model: str,
        port: int,
        host: str = "0.0.0.0",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the command list to start the engine server process locally or in a container."""
        pass

    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate whether the given parameters are compatible with this engine."""
        pass
