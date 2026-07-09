"""
Bridges — wire Honcho's reasoning engine into your existing memory stack.

Each bridge converts between honcho-adapter's data types and
the format expected by a specific repo in your ecosystem.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .deriver import Observation, ObservationSet
from .dialectic import DialecticAgent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# 1. memory-evolution Bridge
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEvolutionEntry:
    """A memory entry compatible with memory-evolution's format."""
    id: str
    content: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    importance_score: float = 1.0
    tags: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)
    version: int = 1
    rewritten_from: str | None = None
    # Honcho-specific
    level: str = "explicit"  # explicit, deductive
    peer_name: str = ""
    session_name: str | None = None
    source_message_ids: list[int] | None = None


class MemoryEvolutionBridge:
    """Bridges Deriver observations into memory-evolution format.

    Usage:
        bridge = MemoryEvolutionBridge("memory_store.json")
        bridge.ingest_observations(obs_set, peer_name="alice", session_name="session-1")
        bridge.run_evolution_cycle()  # runs decay, rewrite, relationship inference
    """

    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path
        self._entries: dict[str, MemoryEvolutionEntry] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            for entry_data in data.get("memories", []):
                entry = MemoryEvolutionEntry(**entry_data)
                self._entries[entry.id] = entry
        except (FileNotFoundError, json.JSONDecodeError):
            self._entries = {}

    def _save(self) -> None:
        import hashlib
        import os
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "memories": [e.__dict__ for e in self._entries.values()],
            }, f, indent=2)

    def _make_id(self, content: str) -> str:
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def ingest_observations(
        self,
        observations: ObservationSet,
        peer_name: str,
        session_name: str | None = None,
    ) -> list[MemoryEvolutionEntry]:
        """Convert Deriver observations to memory-evolution entries and store them."""
        new_entries = []
        now = datetime.now(timezone.utc).timestamp()

        for obs in observations.all:
            entry_id = self._make_id(obs.content)
            if entry_id in self._entries:
                # Update existing — bump version
                existing = self._entries[entry_id]
                existing.access_count += 1
                existing.last_accessed = now
                existing.version += 1
                if session_name and session_name not in (existing.session_name or ""):
                    existing.session_name = session_name
                continue

            entry = MemoryEvolutionEntry(
                id=entry_id,
                content=obs.content,
                timestamp=now,
                tags=[obs.level, peer_name],
                level=obs.level,
                peer_name=peer_name,
                session_name=session_name or obs.session_name,
                source_message_ids=obs.source_ids,
            )
            self._entries[entry_id] = entry
            new_entries.append(entry)

        self._save()
        logger.info(
            "Ingested %d new observations (%d explicit, %d deductive) for %s",
            len(new_entries),
            len(observations.explicit),
            len(observations.deductive),
            peer_name,
        )
        return new_entries

    def query(self, text: str, limit: int = 10) -> list[MemoryEvolutionEntry]:
        """Simple text search over stored observations."""
        text_lower = text.lower()
        scored = []
        for entry in self._entries.values():
            score = entry.importance_score
            if text_lower in entry.content.lower():
                score *= 2.0  # Boost exact matches
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def run_evolution_cycle(self) -> dict[str, Any]:
        """Run decay + relationship inference (simplified memory-evolution)."""
        now = datetime.now(timezone.utc).timestamp()
        decayed = 0
        relationships = 0

        for entry in self._entries.values():
            # Time-based decay: lose 50% relevance every 30 days
            age_days = (now - entry.timestamp) / 86400
            decay = 0.5 ** (age_days / 30.0)
            entry.importance_score = max(0.1, decay)

            if entry.importance_score < 0.2:
                decayed += 1

        # Simple relationship inference: same tags = related
        tag_groups: dict[str, list[str]] = {}
        for eid, entry in self._entries.items():
            for tag in entry.tags:
                tag_groups.setdefault(tag, []).append(eid)

        for tag, ids in tag_groups.items():
            if len(ids) > 1:
                for eid in ids:
                    entry = self._entries[eid]
                    for other_id in ids:
                        if other_id != eid and other_id not in entry.related_memories:
                            entry.related_memories.append(other_id)
                            relationships += 1

        self._save()
        return {
            "memories_processed": len(self._entries),
            "memories_decayed": decayed,
            "relationships_inferred": relationships,
        }

    def get_observations_by_peer(self, peer_name: str) -> list[MemoryEvolutionEntry]:
        """Get all observations for a specific peer."""
        return [e for e in self._entries.values() if e.peer_name == peer_name]

    def get_observations_by_level(self, level: str) -> list[MemoryEvolutionEntry]:
        """Get all observations of a specific level (explicit/deductive)."""
        return [e for e in self._entries.values() if e.level == level]


# ══════════════════════════════════════════════════════════════════════════
# 2. OBLIVIARCH Bridge
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ObliviarchTrace:
    """A trace compatible with OBLIVIARCH's episodic format."""
    agents: list[str]
    task: str
    steps: list[dict[str, str]]
    outcome: str = "success"
    score: float = 1.0
    observations: list[str] = field(default_factory=list)


class ObliviarchBridge:
    """Feeds deriver observations into OBLIVIARCH's compression pipeline.

    Observations become episodic traces → promoted to schemas → ascend to archetypes.
    """

    @staticmethod
    def observation_to_trace(
        obs: Observation,
        peer_name: str,
        session_name: str | None = None,
    ) -> ObliviarchTrace:
        """Convert a single observation to an OBLIVIARCH trace."""
        return ObliviarchTrace(
            agents=[peer_name],
            task=f"observation: {obs.level}",
            steps=[
                {"action": "OBSERVE", "agent": "deriver"},
                {"action": "EXTRACT", "agent": "deriver"},
                {"action": "STORE", "agent": "memory-evolution"},
            ],
            outcome="success",
            score=1.0,
            observations=[obs.content],
        )

    @staticmethod
    def observations_to_traces(
        observations: ObservationSet,
        peer_name: str,
        session_name: str | None = None,
    ) -> list[ObliviarchTrace]:
        """Convert an ObservationSet to a list of OBLIVIARCH traces."""
        return [
            ObliviarchBridge.observation_to_trace(obs, peer_name, session_name)
            for obs in observations.all
        ]

    @staticmethod
    def format_for_obliviarch_record(observations: ObservationSet, peer_name: str) -> dict[str, Any]:
        """Format observations for OBLIVIARCH's record() API."""
        return {
            "agents": [peer_name, "deriver"],
            "task": f"observation extraction for {peer_name}",
            "steps": [
                {"action": "EXTRACT", "agent": "deriver"},
                {"action": "CLASSIFY", "agent": "deriver"},
                {"action": "STORE", "agent": "memory-evolution"},
            ]
            + [
                {"action": "OBSERVATION", "agent": peer_name, "detail": obs.content[:200]}
                for obs in observations.all
            ],
            "outcome": "success",
            "score": observations.count / max(observations.count, 1),
        }


# ══════════════════════════════════════════════════════════════════════════
# 3. Deterministic-Retrieval Bridge
# ══════════════════════════════════════════════════════════════════════════

class DeterministicRetrievalBridge:
    """Makes observations accessible via deterministic-retrieval's path-based lookup.

    Observations are stored as JSON files in a structured directory:
        observations/{peer_name}/{level}/{yyyymmdd}.json
    """

    def __init__(self, base_path: str = "~/.hermes/workspace/observations"):
        import os
        self.base_path = os.path.expanduser(base_path)

    def _ensure_dir(self, peer_name: str, level: str, date_str: str) -> str:
        import os
        path = os.path.join(self.base_path, peer_name, level)
        os.makedirs(path, exist_ok=True)
        return path

    def store_observations(
        self,
        observations: ObservationSet,
        peer_name: str,
    ) -> dict[str, str]:
        """Store observations as JSON files, return file paths."""
        import os
        from datetime import datetime

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        result = {}

        for level_name, obs_list in [
            ("explicit", observations.explicit),
            ("deductive", observations.deductive),
        ]:
            if not obs_list:
                continue
            dir_path = self._ensure_dir(peer_name, level_name, date_str)
            file_path = os.path.join(dir_path, f"{date_str}.json")

            # Load existing or create new
            try:
                with open(file_path) as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []

            existing.extend([
                {"content": o.content, "timestamp": o.created_at or datetime.now(timezone.utc).isoformat()}
                for o in obs_list
            ])

            with open(file_path, "w") as f:
                json.dump(existing, f, indent=2)

            result[level_name] = file_path

        return result

    def get_path_for(self, peer_name: str, level: str = "explicit") -> str:
        """Get the deterministic path for a peer's observations."""
        import os
        return os.path.join(self.base_path, peer_name, level)


# ══════════════════════════════════════════════════════════════════════════
# 4. Recursive-Retrieval Bridge
# ══════════════════════════════════════════════════════════════════════════

class RecursiveRetrievalBridge:
    """Makes observations available as a context layer in recursive-retrieval.

    Observations sit at the agent layer (L1) in the tree:
        L0: Shared (global context)
        L1: Agent (observations about this peer)
        L2: Session (session-specific context)
    """

    @staticmethod
    def observations_to_agent_layer(
        peer_name: str,
        explicit: list[Observation],
        deductive: list[Observation] | None = None,
    ) -> dict[str, Any]:
        """Convert observations to an agent-layer dict for recursive-retrieval."""
        layer = {
            "peer_name": peer_name,
            "observations": {
                "explicit": [o.content for o in explicit],
                "deductive": [o.content for o in (deductive or [])],
            },
            "observation_count": len(explicit) + len(deductive or []),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        return layer

    @staticmethod
    def observations_to_session_layer(
        session_name: str,
        observations: ObservationSet,
    ) -> dict[str, Any]:
        """Convert session observations to a session-layer dict."""
        return {
            "session_name": session_name,
            "observations": observations.to_dict(),
            "observation_count": observations.count,
        }


# ══════════════════════════════════════════════════════════════════════════
# 5. Hermes-Namespace Bridge
# ══════════════════════════════════════════════════════════════════════════

class NamespaceBridge:
    """Registers observation URI handlers for hermes-namespace.

    URI format:
        hermes://observations/{peer_name}/{level}/{date}
        hermes://observations/{peer_name}  (latest all)
    """

    def __init__(self, base_path: str = "~/.hermes/workspace/observations"):
        import os
        self.base_path = os.path.expanduser(base_path)

    def resolve(self, uri: str) -> str:
        """Resolve a hermes://observations/... URI to a file path.

        Args:
            uri: e.g. hermes://observations/alice/explicit/20260709

        Returns:
            Resolved file path.
        """
        import os
        import re

        # Parse: hermes://observations/{peer_name}/{level}/{date}
        match = re.match(
            r"hermes://observations/([^/]+)(?:/(explicit|deductive))?(?:/(\d{8}))?",
            uri,
        )
        if not match:
            raise ValueError(f"Invalid observation URI: {uri}")

        peer_name = match.group(1)
        level = match.group(2) or "explicit"
        date_str = match.group(3)

        if date_str:
            return os.path.join(self.base_path, peer_name, level, f"{date_str}.json")
        else:
            # Return directory
            return os.path.join(self.base_path, peer_name, level)

    def register_with_namespace(self) -> dict[str, str]:
        """Return handler registration info for hermes-namespace.

        Usage:
            # In hermes-namespace config:
            handlers = namespace_bridge.register_with_namespace()
            # Add to resolver's handler registry
        """
        return {
            "namespace": "observations",
            "description": "Access deriver observations by peer, level, and date",
            "example": "hermes://observations/alice/explicit/20260709",
            "resolve": "NamespaceBridge.resolve()",
        }


# ══════════════════════════════════════════════════════════════════════════
# 6. NEUROSWARM Bridge
# ══════════════════════════════════════════════════════════════════════════

class NeuroSwarmBridge:
    """Wires the dialectic agent into NEUROSWARM's dual-phase dispatch.

    Phase 1 (WHAT): GBrain decides what to investigate.
    Phase 2 (HOW): Dialectic agent answers the question using memory tools.
    """

    @staticmethod
    def create_dialectic_tool(agent: DialecticAgent) -> dict[str, Any]:
        """Wrap a DialecticAgent as a NEUROSWARM tool.

        Returns a tool descriptor compatible with NEUROSWARM's tool registry.
        """
        return {
            "name": "query_peer_memory",
            "description": f"Answer questions about {agent.observed} from {agent.observer}'s perspective",
            "fn": agent.answer,
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Natural language question about the peer",
                }
            },
        }

    @staticmethod
    def format_for_council_context(
        peer_name: str,
        observations: ObservationSet,
        max_obs: int = 20,
    ) -> str:
        """Format observations as context for NEUROSWARM's 10-D council.

        The council uses this context to deliberate on what to do next.
        """
        parts = [f"## Observations about {peer_name}", ""]
        parts.append(f"Total: {observations.count} ({len(observations.explicit)} explicit, {len(observations.deductive)} deductive)")
        parts.append("")

        for obs in observations.all[:max_obs]:
            tag = "📌" if obs.level == "explicit" else "🧠"
            parts.append(f"{tag} [{obs.level}] {obs.content}")

        if observations.count > max_obs:
            parts.append(f"\n... and {observations.count - max_obs} more observations")

        return "\n".join(parts)

    @staticmethod
    def council_verdict_to_observations(
        verdict: str,
        peer_name: str,
        session_name: str | None = None,
    ) -> ObservationSet:
        """Convert a NEUROSWARM council verdict into new deductive observations.

        The council may identify patterns, contradictions, or insights that
        should be stored as deductive observations.
        """
        return ObservationSet(
            deductive=[
                Observation(
                    content=verdict,
                    level="deductive",
                    peer_name=peer_name,
                    session_name=session_name,
                )
            ]
        )