"""Anthropic integration package (deterministic shape + LLM hydration)."""

from .client import AnthropicTopicCompiler, get_anthropic_topic_compiler

__all__ = ["AnthropicTopicCompiler", "get_anthropic_topic_compiler"]
