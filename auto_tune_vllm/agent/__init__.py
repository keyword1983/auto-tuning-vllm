"""Claude-driven vLLM performance tuning agent.

Install optional dependencies with::

    pip install -e ".[agent]"

Run via::

    auto-tune-vllm agent --vllm-endpoint http://localhost:8000 --model <model> ...
    python -m auto_tune_vllm.agent ...
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
