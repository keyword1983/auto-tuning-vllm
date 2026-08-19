"""
LLM wrapper with token tracking for Claude API (direct Anthropic and Vertex AI).
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic
from anthropic import AnthropicVertex

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


def to_vertex_model_id(model_id: str) -> str:
    """Convert a standard Anthropic model ID to Vertex AI format.

    Vertex AI uses '@' instead of the last '-' before the date stamp.
    e.g. "claude-sonnet-4-20250514" -> "claude-sonnet-4@20250514"
         "claude-opus-4-20250514"   -> "claude-opus-4@20250514"
         "claude-haiku-4-5-20251001" -> "claude-haiku-4-5@20251001"
    If the model already contains '@', return as-is.
    """
    if "@" in model_id:
        return model_id
    # Match the last '-' followed by a pure-digit date suffix
    match = re.match(r"^(.+)-(\d{8,})$", model_id)
    if match:
        return f"{match.group(1)}@{match.group(2)}"
    return model_id


@dataclass
class TokenUsage:
    """Track token usage per model."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_creation_tokens += cache_creation_tokens
        self.calls += 1

    def cost(self) -> float:
        """Calculate cost in USD (accounts for cache discount)."""
        pricing = MODEL_PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        # Cache read tokens cost 90% less, cache creation costs 25% more
        regular_input = (
            self.input_tokens - self.cache_read_tokens - self.cache_creation_tokens
        )
        input_cost = (regular_input / 1_000_000) * pricing["input"]
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * pricing["input"] * 0.1
        cache_create_cost = (
            (self.cache_creation_tokens / 1_000_000) * pricing["input"] * 1.25
        )
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + cache_read_cost + cache_create_cost + output_cost

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "api_calls": self.calls,
            "cost_usd": round(self.cost(), 4),
        }


@dataclass
class LLMResponse:
    """Response from LLM with content and token info."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str


# Available models
MODELS = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "haiku": "claude-haiku-4-5-20251001",
}


def get_model_id(model_name: str) -> str:
    """Convert short model name to full model ID."""
    if model_name in MODELS:
        return MODELS[model_name]
    return model_name


class ClaudeClient:
    """Claude API client with token tracking (direct Anthropic API or Vertex AI)."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_vertex: bool = False,
        vertex_project_id: Optional[str] = None,
        vertex_region: str = "us-east5",
    ):
        self.use_vertex = use_vertex
        self.model = model or self.DEFAULT_MODEL
        self.usage: dict[str, TokenUsage] = {}

        if self.use_vertex:
            project_id = vertex_project_id or os.environ.get(
                "ANTHROPIC_VERTEX_PROJECT_ID"
            )
            if not project_id:
                raise ValueError(
                    "Vertex AI requires a project ID. "
                    "Set --vertex-project-id or ANTHROPIC_VERTEX_PROJECT_ID."
                )
            region = vertex_region or os.environ.get("CLOUD_ML_REGION", "us-east5")
            self.client = AnthropicVertex(
                project_id=project_id,
                region=region,
            )
            # Convert model ID to Vertex format (uses '@' separator)
            self.model = to_vertex_model_id(self.model)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    def _get_usage(self, model: str) -> TokenUsage:
        if model not in self.usage:
            self.usage[model] = TokenUsage(model=model)
        return self.usage[model]

    def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt to Claude and track token usage."""
        model = model or self.model

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Track usage
        self._get_usage(model).add(input_tokens, output_tokens)

        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    def get_total_usage(self) -> list[dict]:
        """Get token usage summary for all models."""
        return [usage.to_dict() for usage in self.usage.values()]

    def get_usage_data(self) -> list:
        """Return token usage as a list of dicts for structured reporting."""
        data = []
        for usage in self.usage.values():
            data.append(
                {
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "api_calls": usage.calls,
                    "cost_usd": usage.cost(),
                }
            )
        return data

    def get_usage_report(self) -> str:
        """Generate a markdown table of token usage with cost."""
        lines = [
            "| Model | Input Tokens | Output Tokens | Total | API Calls | Cost (USD) |",
            "|-------|--------------|---------------|-------|-----------|------------|",
        ]

        total_input = 0
        total_output = 0
        total_calls = 0
        total_cost = 0.0

        for usage in self.usage.values():
            total = usage.input_tokens + usage.output_tokens
            cost = usage.cost()
            lines.append(
                f"| {usage.model} | {usage.input_tokens:,} | {usage.output_tokens:,} | {total:,} | {usage.calls} | ${cost:.4f} |"
            )
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_calls += usage.calls
            total_cost += cost

        lines.append(
            f"| **Total** | **{total_input:,}** | **{total_output:,}** | **{total_input + total_output:,}** | **{total_calls}** | **${total_cost:.4f}** |"
        )

        return "\n".join(lines)
