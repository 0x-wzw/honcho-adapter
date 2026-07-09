"""
Deriver — LLM-based observation extraction from conversations.

Adapted from Honcho's src/deriver/ module. Stripped of Postgres/server deps.
Extracts explicit + deductive observations from raw conversation text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


# ── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A single observation extracted from a conversation."""
    content: str
    level: str  # "explicit" or "deductive"
    peer_name: str
    session_name: str | None = None
    created_at: str | None = None
    source_ids: list[int] | None = None  # message IDs this was derived from
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationSet:
    """Collection of observations from a single deriver pass."""
    explicit: list[Observation] = field(default_factory=list)
    deductive: list[Observation] = field(default_factory=list)

    @property
    def all(self) -> list[Observation]:
        return self.explicit + self.deductive

    @property
    def count(self) -> int:
        return len(self.explicit) + len(self.deductive)

    def is_empty(self) -> bool:
        return self.count == 0

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "explicit": [o.__dict__ for o in self.explicit],
            "deductive": [o.__dict__ for o in self.deductive],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationSet:
        return cls(
            explicit=[Observation(**o) for o in data.get("explicit", [])],
            deductive=[Observation(**o) for o in data.get("deductive", [])],
        )


# ── Deriver ────────────────────────────────────────────────────────────────

@dataclass
class DeriverConfig:
    """Configuration for the deriver."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    max_input_tokens: int | None = None
    temperature: float = 0.0
    custom_instructions: str | None = None
    retry_attempts: int = 3


DEFAULT_DERIVER_PROMPT = """\
Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

[DEDUCTIVE] DEFINITION: Inferences or conclusions that go beyond what is explicitly stated.
   - Combine multiple explicit observations to form a higher-level insight
   - These should be labeled as "deductive" in the output

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Use the exact peer id from `Target peer:` in final observations.
- Properly attribute observations to the correct subject.
- Observations should make sense on their own.
- Extract ALL observations from the target peer's messages.
- Contextualize each observation sufficiently.

OUTPUT FORMAT:
Return a JSON object with two keys:
- "explicit": list of strings, each an explicit observation
- "deductive": list of strings, each a deductive observation

Example:
{
  "explicit": [
    "alice is 25 years old",
    "alice's birthday is June 21st",
    "alice has a dog",
    "alice lives in NYC"
  ],
  "deductive": [
    "alice completed high school or equivalent",
    "alice enjoys outdoor activities"
  ]
}

{peer_section}
{messages_section}
"""


class Deriver:
    """Extracts observations from conversation text using an LLM.

    Usage:
        deriver = Deriver(llm_backend)
        obs = await deriver.extract(
            peer_name="alice",
            messages=[
                {"peer": "alice", "content": "I just turned 25!", "timestamp": "..."},
                {"peer": "bob", "content": "Happy birthday!", "timestamp": "..."},
            ],
        )
        print(obs.explicit)  # ["alice is 25 years old"]
    """

    def __init__(
        self,
        llm: LLMBackend,
        config: DeriverConfig | None = None,
    ):
        self._llm = llm
        self.config = config or DeriverConfig()

    async def extract(
        self,
        peer_name: str,
        messages: list[dict[str, Any]],
        custom_instructions: str | None = None,
    ) -> ObservationSet:
        """Extract observations from a list of messages about a peer.

        Args:
            peer_name: The peer being analyzed (the target).
            messages: List of dicts with keys: peer, content, timestamp (optional).
            custom_instructions: Optional instructions to bias extraction.

        Returns:
            ObservationSet with explicit and deductive observations.
        """
        # Format messages
        formatted = self._format_messages(messages)

        # Build prompt
        ci = custom_instructions or self.config.custom_instructions
        peer_section = f"Target peer:\n{peer_name}\n"
        if ci:
            peer_section += f"\nCUSTOM INSTRUCTIONS:\n{ci}\n"
        messages_section = f"Messages to analyze:\n<messages>\n{formatted}\n</messages>"

        prompt = DEFAULT_DERIVER_PROMPT.format(
            peer_section=peer_section,
            messages_section=messages_section,
        )

        # Call LLM
        response = await self._llm.call(
            prompt=prompt,
            json_mode=True,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            retry_attempts=self.config.retry_attempts,
        )

        # Parse JSON response
        obs = self._parse_response(response.content, peer_name)
        if obs.is_empty():
            logger.warning("Deriver produced zero observations for %s", peer_name)

        return obs

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the prompt."""
        parts = []
        for msg in messages:
            peer = msg.get("peer", "unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            if ts:
                parts.append(f"[{ts}] {peer}: {content}")
            else:
                parts.append(f"{peer}: {content}")
        return "\n".join(parts)

    def _parse_response(self, content: str, peer_name: str) -> ObservationSet:
        """Parse the LLM JSON response into an ObservationSet."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse deriver JSON response: %s", content[:200])
            return ObservationSet()

        explicit_raw = data.get("explicit", [])
        deductive_raw = data.get("deductive", [])

        explicit = [
            Observation(content=obs, level="explicit", peer_name=peer_name)
            for obs in explicit_raw if isinstance(obs, str)
        ]
        deductive = [
            Observation(content=obs, level="deductive", peer_name=peer_name)
            for obs in deductive_raw if isinstance(obs, str)
        ]

        return ObservationSet(explicit=explicit, deductive=deductive)


# ── Convenience ────────────────────────────────────────────────────────────

def create_deriver(
    provider: str = "openai",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> Deriver:
    """Factory: create a Deriver with a single call."""
    from .llm import create_backend

    config = DeriverConfig(model=model, **{k: v for k, v in kwargs.items() if k in
        ["max_tokens", "temperature", "custom_instructions", "retry_attempts"]})
    llm = create_backend(provider=provider, api_key=api_key)
    return Deriver(llm=llm, config=config)