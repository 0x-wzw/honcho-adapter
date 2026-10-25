"""
Deriver — LLM-based observation extraction from conversations.

Adapted from Honcho's src/deriver/ module. Stripped of Postgres/server deps.
Extracts explicit + deductive observations from raw conversation text.

v2.0 — Improved performance over time:
- Confidence scoring per observation (0.0-1.0)
- Contradiction detection within a single extraction pass
- Source tracking (which messages produced which observations)
- Dedup within a single pass (same observation from multiple messages)
- Custom instructions for domain-specific extraction (e.g., financial analysis)
- Structured metadata extraction (entities, relationships, sentiment)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


# ── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A single observation extracted from a conversation.
    
    v2: Added confidence, contradiction_flags, entities, sentiment.
    """
    content: str
    level: str  # "explicit" or "deductive"
    peer_name: str
    session_name: str | None = None
    created_at: str | None = None
    source_ids: list[int] | None = None  # message IDs this was derived from
    metadata: dict[str, Any] = field(default_factory=dict)
    # v2 improvements
    confidence: float = 0.9  # 0.0-1.0
    contradiction_flags: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)  # [{"name": "...", "type": "person|org|place|metric"}]
    sentiment: str = "neutral"  # positive, negative, neutral, mixed


@dataclass
class ObservationSet:
    """Collection of observations from a single deriver pass.
    
    v2: Deduplicates within the set, detects internal contradictions.
    """
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

    def deduplicate(self) -> None:
        """Remove duplicate observations within this set.
        
        v2: New method. Uses content hash for exact dedup.
        """
        seen: set[str] = set()
        self.explicit = [o for o in self.explicit if not (o.content in seen or seen.add(o.content))]
        seen.clear()
        self.deductive = [o for o in self.deductive if not (o.content in seen or seen.add(o.content))]

    def detect_internal_contradictions(self) -> list[tuple[str, str]]:
        """Detect contradictions between observations within this set.
        
        v2: New method. Returns list of (obs_a, obs_b) pairs that contradict.
        """
        contradictions = []
        all_obs = self.all
        negations = {"not", "never", "no", "doesn't", "don't", "didn't", "won't", "can't"}
        opp_pairs = [("increased", "decreased"), ("rose", "fell"), ("higher", "lower"),
                     ("better", "worse"), ("positive", "negative"), ("approved", "rejected")]
        
        for i, a in enumerate(all_obs):
            a_lower = a.content.lower()
            a_words = set(re.findall(r'\b[a-z]+\b', a_lower))
            a_has_neg = any(n in a_words for n in negations)
            
            for b in all_obs[i+1:]:
                b_lower = b.content.lower()
                b_words = set(re.findall(r'\b[a-z]+\b', b_lower))
                b_has_neg = any(n in b_words for n in negations)
                
                # Check negation-based contradiction
                if a_has_neg != b_has_neg:
                    a_content = a_words - negations
                    b_content = b_words - negations
                    if len(a_content & b_content) >= 2:
                        contradictions.append((a.content, b.content))
                        a.contradiction_flags.append(b.content[:80])
                        b.contradiction_flags.append(a.content[:80])
                        continue
                
                # Check opposite-direction contradiction
                for o1, o2 in opp_pairs:
                    if (o1 in a_lower and o2 in b_lower) or (o2 in a_lower and o1 in b_lower):
                        contradictions.append((a.content, b.content))
                        a.contradiction_flags.append(b.content[:80])
                        b.contradiction_flags.append(a.content[:80])
                        break
        
        return contradictions


# ── Deriver ────────────────────────────────────────────────────────────────

@dataclass
class DeriverConfig:
    """Configuration for the deriver.
    
    v2: Added confidence_threshold, domain_context, extract_entities.
    """
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    max_input_tokens: int | None = None
    temperature: float = 0.0
    custom_instructions: str | None = None
    retry_attempts: int = 3
    # v2 improvements
    confidence_threshold: float = 0.3  # Minimum confidence to keep an observation
    extract_entities: bool = True  # Extract named entities from observations
    detect_sentiment: bool = True  # Detect sentiment per observation
    domain_context: str | None = None  # e.g., "financial analysis", "software engineering"


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
- Assign a confidence score (0.0-1.0) to each observation based on how certain the evidence is.
- Flag any observations that appear to contradict each other.

OUTPUT FORMAT:
Return a JSON object with the following keys:
- "explicit": list of objects, each with "content" (string), "confidence" (float 0-1), "entities" (list of {{name, type}}), "sentiment" (string)
- "deductive": list of objects, each with "content" (string), "confidence" (float 0-1), "entities" (list of {{name, type}}), "sentiment" (string)
- "contradictions": list of [string, string] pairs where observations contradict each other

Example:
{{
  "explicit": [
    {{"content": "alice is 25 years old", "confidence": 1.0, "entities": [{{"name": "alice", "type": "person"}}], "sentiment": "neutral"}},
    {{"content": "alice has a dog", "confidence": 0.9, "entities": [{{"name": "alice", "type": "person"}}], "sentiment": "positive"}}
  ],
  "deductive": [
    {{"content": "alice completed high school or equivalent", "confidence": 0.7, "entities": [{{"name": "alice", "type": "person"}}], "sentiment": "neutral"}}
  ],
  "contradictions": []
}}

{peer_section}
{messages_section}
"""


class Deriver:
    """Extracts observations from conversation text using an LLM.
    
    v2: Confidence scoring, contradiction detection, entity extraction, sentiment.
    
    Usage:
        deriver = Deriver(llm_backend)
        obs = await deriver.extract(
            peer_name="alice",
            messages=[...],
        )
        print(obs.explicit)  # [Observation(...), ...]
        print(obs.detect_internal_contradictions())  # [(content_a, content_b), ...]
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
        
        v2: Returns ObservationSet with confidence scores, entities, sentiment,
        and internal contradiction detection.
        
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
        domain = self.config.domain_context
        
        peer_section = f"Target peer:\n{peer_name}\n"
        if ci:
            peer_section += f"\nCUSTOM INSTRUCTIONS:\n{ci}\n"
        if domain:
            peer_section += f"\nDOMAIN CONTEXT:\n{domain}\n"
        
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
        obs = self._parse_response(response.content, peer_name, messages)
        
        # Post-processing
        obs.deduplicate()
        contradictions = obs.detect_internal_contradictions()
        
        if contradictions:
            logger.info(
                "Detected %d contradictions in extraction for %s",
                len(contradictions), peer_name,
            )
        
        if obs.is_empty():
            logger.warning("Deriver produced zero observations for %s", peer_name)

        return obs

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the prompt."""
        parts = []
        for i, msg in enumerate(messages):
            peer = msg.get("peer", "unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp", "")
            if ts:
                parts.append(f"[{ts}] [{i}] {peer}: {content}")
            else:
                parts.append(f"[{i}] {peer}: {content}")
        return "\n".join(parts)

    def _parse_response(
        self, 
        content: str, 
        peer_name: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> ObservationSet:
        """Parse the LLM JSON response into an ObservationSet.
        
        v2: Handles structured observation format with confidence, entities, sentiment.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse deriver JSON response: %s", content[:200])
            return ObservationSet()

        explicit_raw = data.get("explicit", [])
        deductive_raw = data.get("deductive", [])
        contradictions_raw = data.get("contradictions", [])

        def _parse_obs(raw: Any, level: str) -> list[Observation]:
            if isinstance(raw, str):
                # Simple string format (backward compat)
                return [
                    Observation(
                        content=raw,
                        level=level,
                        peer_name=peer_name,
                        confidence=0.9 if level == "explicit" else 0.7,
                    )
                ]
            elif isinstance(raw, dict):
                # Structured format with confidence, entities, sentiment
                return [
                    Observation(
                        content=raw.get("content", str(raw)),
                        level=level,
                        peer_name=peer_name,
                        confidence=raw.get("confidence", 0.9 if level == "explicit" else 0.7),
                        entities=raw.get("entities", []),
                        sentiment=raw.get("sentiment", "neutral"),
                    )
                ]
            return []

        explicit = []
        for raw in explicit_raw:
            explicit.extend(_parse_obs(raw, "explicit"))

        deductive = []
        for raw in deductive_raw:
            deductive.extend(_parse_obs(raw, "deductive"))

        # Filter by confidence threshold
        explicit = [o for o in explicit if o.confidence >= self.config.confidence_threshold]
        deductive = [o for o in deductive if o.confidence >= self.config.confidence_threshold]

        # Map source message IDs
        if messages:
            for i, msg in enumerate(messages):
                msg_content = msg.get("content", "").lower()
                for obs in explicit + deductive:
                    # Simple heuristic: if observation content shares words with message
                    obs_words = set(obs.content.lower().split())
                    msg_words = set(msg_content.split())
                    if len(obs_words & msg_words) >= 2:
                        if obs.source_ids is None:
                            obs.source_ids = []
                        if i not in obs.source_ids:
                            obs.source_ids.append(i)

        obs_set = ObservationSet(explicit=explicit, deductive=deductive)
        
        # Add contradiction flags from LLM output
        if contradictions_raw:
            for pair in contradictions_raw:
                if isinstance(pair, list) and len(pair) == 2:
                    for obs in obs_set.all:
                        if obs.content == pair[0]:
                            obs.contradiction_flags.append(pair[1][:80])
                        elif obs.content == pair[1]:
                            obs.contradiction_flags.append(pair[0][:80])

        return obs_set


# ── Convenience ────────────────────────────────────────────────────────────

def create_deriver(
    provider: str = "openai",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> Deriver:
    """Factory: create a Deriver with a single call.
    
    v2: Accepts domain_context, confidence_threshold, extract_entities kwargs.
    """
    from .llm import create_backend

    valid_keys = [
        "max_tokens", "temperature", "custom_instructions", "retry_attempts",
        "confidence_threshold", "extract_entities", "detect_sentiment", "domain_context",
    ]
    config = DeriverConfig(
        model=model,
        **{k: v for k, v in kwargs.items() if k in valid_keys},
    )
    llm = create_backend(provider=provider, api_key=api_key)
    return Deriver(llm=llm, config=config)
