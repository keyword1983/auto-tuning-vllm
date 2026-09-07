"""Factory function for retrieving engine adapters."""

from __future__ import annotations

import logging
from typing import Optional

from .base import EngineAdapter
from .vllm import VLLMEngineAdapter
from .sglang import SGLangEngineAdapter
from .llamacpp import LlamaCppEngineAdapter

logger = logging.getLogger(__name__)

_ADAPTERS = {
    "vllm": VLLMEngineAdapter,
    "sglang": SGLangEngineAdapter,
    "llamacpp": LlamaCppEngineAdapter,
}


def get_engine_adapter(engine_type: Optional[str] = None) -> EngineAdapter:
    """Retrieve the corresponding EngineAdapter for a given engine type name.

    Supported engines: 'vllm', 'sglang', 'llamacpp' (or 'llama.cpp').
    Defaults to 'vllm' if omitted or unknown.
    """
    if not engine_type:
        return VLLMEngineAdapter()

    normalized = (
        str(engine_type)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )

    if normalized in ("vllm",):
        return VLLMEngineAdapter()
    elif normalized in ("sglang",):
        return SGLangEngineAdapter()
    elif normalized in ("llamacpp", "llama"):
        return LlamaCppEngineAdapter()

    logger.warning(
        "Unknown engine type '%s', falling back to default vLLM adapter",
        engine_type,
    )
    return VLLMEngineAdapter()
