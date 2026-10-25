"""
honcho-adapter — Honcho's reasoning-first memory, extracted and bridged into your existing Hermes stack.

No Postgres. No server. No AGPL. Just the core reasoning engine:
- Deriver: LLM-based observation extraction from conversations
- Dialectic: Agentic query answering with tool-based context gathering
- Surprisal: Novelty detection for finding interesting/contradictory observations
"""

from __future__ import annotations

__version__ = "2.0.1"

from .deriver import Deriver, DeriverConfig, Observation, ObservationSet
from .dialectic import DialecticAgent, DialecticConfig
from .surprisal import SurprisalEngine, SurprisalConfig, SurprisalScore
from .llm import LLMBackend, LLMConfig, OpenAIProvider, AnthropicProvider, GeminiProvider

__all__ = [
    "Deriver", "DeriverConfig", "Observation", "ObservationSet",
    "DialecticAgent", "DialecticConfig",
    "SurprisalEngine", "SurprisalConfig", "SurprisalScore",
    "LLMBackend", "LLMConfig", "OpenAIProvider", "AnthropicProvider", "GeminiProvider",
]