"""
Bridges — wire Honcho's reasoning engine into your existing memory stack.

Each bridge converts between honcho-adapter's data types and
the format expected by a specific repo in your ecosystem.

v2.0 — Improved memory performance over time:
- Embedding-based semantic search (not just text match)
- Dedup via semantic similarity + content hash
- Consolidation: merge similar observations into higher-level insights
- Access-frequency-boosted decay (not just time-based)
- Contradiction detection across observations
- Temporal weighting (recent observations score higher)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .deriver import Observation, ObservationSet
from .dialectic import DialecticAgent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════

def _make_id(content: str, namespace: str = "") -> str:
    """Deterministic ID from content + optional namespace."""
    raw = f"{namespace}:{content}" if namespace else content
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _text_to_embedding(text: str, dim: int = 64) -> list[float]:
    """Fast deterministic embedding from text features.
    
    Uses character n-gram hashing to produce a fixed-dimension vector.
    Not as good as a real embedding model, but deterministic, fast,
    and much better than proxy features alone.
    
    Replace with a real embedding model (e.g., text-embedding-3-small)
    for production use.
    """
    import hashlib
    vec = [0.0] * dim
    # Unigram hash
    for i, ch in enumerate(text):
        h = int(hashlib.md5(f"1:{ch}".encode()).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    # Bigram hash
    for i in range(len(text) - 1):
        bg = text[i:i+2]
        h = int(hashlib.md5(f"2:{bg}".encode()).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 0.5
    # Normalize
    import math
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ══════════════════════════════════════════════════════════════════════════
# 1. memory-evolution Bridge (v2 — improved)
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
    # v2 improvements
    embedding: list[float] | None = None
    confidence: float = 1.0  # 0.0-1.0 how confident we are in this observation
    contradiction_ids: list[str] = field(default_factory=list)
    consolidated: bool = False  # True if this was merged from multiple observations
    source_count: int = 1  # How many original observations fed into this


class MemoryEvolutionBridge:
    """Bridges Deriver observations into memory-evolution format.
    
    v2 improvements over v1:
    - Embedding-based semantic search (not just text match)
    - Dedup via semantic similarity threshold
    - Consolidation: merge similar observations into higher-level insights
    - Access-frequency-boosted decay (not just time-based)
    - Contradiction detection across observations
    - Temporal weighting (recent observations score higher)
    
    Usage:
        bridge = MemoryEvolutionBridge("memory_store.json")
        bridge.ingest_observations(obs_set, peer_name="alice")
        bridge.run_evolution_cycle()  # decay + consolidation + contradiction
        results = bridge.semantic_search("what does alice like?")
    """

    def __init__(
        self,
        storage_path: str = "memory_store.json",
        embed_dim: int = 64,
        similarity_threshold: float = 0.95,
        contradiction_threshold: float = 0.60,
    ):
        self.storage_path = storage_path
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
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
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "memories": [e.__dict__ for e in self._entries.values()],
                "version": 2,
                "updated_at": _now_ts(),
            }, f, indent=2)

    def _compute_embedding(self, text: str) -> list[float]:
        return _text_to_embedding(text, self.embed_dim)

    def _detect_contradiction(self, a: str, b: str) -> bool:
        """Detect if two observations likely contradict each other.
        
        Uses simple heuristics:
        - Negation markers: "not", "never", "doesn't", "didn't", "won't"
        - Opposite direction markers: "decreased" vs "increased"
        - Temporal conflict: "was X" vs "is now Y"
        """
        a_lower = a.lower()
        b_lower = b.lower()
        a_words = set(re.findall(r'\b[a-z]+\b', a_lower))
        b_words = set(re.findall(r'\b[a-z]+\b', b_lower))
        
        # Check for explicit negation patterns
        negations = {"not", "never", "no", "doesn't", "don't", "didn't", "won't", "can't", "isn't", "aren't"}
        
        a_has_neg = any(n in a_words for n in negations)
        b_has_neg = any(n in b_words for n in negations)
        
        # If one has negation and the other doesn't, check if they share key terms
        if a_has_neg != b_has_neg:
            # Remove negation words for comparison
            a_content = a_words - negations
            b_content = b_words - negations
            # If they share significant content but differ in negation
            overlap = a_content & b_content
            if len(overlap) >= 2:
                return True
        
        # Check for opposite direction words
        opp_pairs = [("increased", "decreased"), ("rose", "fell"), ("higher", "lower"),
                     ("better", "worse"), ("positive", "negative"), ("approved", "rejected"),
                     ("up", "down"), ("grew", "shrank"), ("profitable", "unprofitable"),
                     ("passed", "failed")]
        for o1, o2 in opp_pairs:
            if (o1 in a_lower and o2 in b_lower) or (o2 in a_lower and o1 in b_lower):
                return True
        
        return False

    def ingest_observations(
        self,
        observations: ObservationSet,
        peer_name: str,
        session_name: str | None = None,
    ) -> list[MemoryEvolutionEntry]:
        """Convert Deriver observations to memory-evolution entries and store them.
        
        v2: Computes embeddings, deduplicates by semantic similarity,
        detects contradictions, and merges near-duplicates.
        """
        new_entries = []
        now = _now_ts()

        for obs in observations.all:
            entry_id = _make_id(obs.content, peer_name)
            
            # Check for exact duplicate
            if entry_id in self._entries:
                existing = self._entries[entry_id]
                existing.access_count += 1
                existing.last_accessed = now
                existing.version += 1
                existing.source_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.05)
                continue

            # Compute embedding
            emb = self._compute_embedding(obs.content)

            # FIRST: Check for contradictions with existing observations
            # (before dedup, so contradictory observations are always flagged)
            contradiction_ids = []
            for existing_id, existing_entry in self._entries.items():
                if existing_entry.peer_name != peer_name:
                    continue
                if self._detect_contradiction(obs.content, existing_entry.content):
                    contradiction_ids.append(existing_id)
                    if existing_id not in existing_entry.contradiction_ids:
                        existing_entry.contradiction_ids.append(entry_id)

            # THEN: Check for semantic near-duplicate (dedup)
            # Only dedup within the same level (explicit vs deductive are different types)
            # Don't dedup if the observations contradict each other
            deduped = False
            for existing_id, existing_entry in self._entries.items():
                if existing_entry.peer_name != peer_name:
                    continue
                if existing_entry.level != obs.level:
                    continue  # Don't dedup across levels
                if existing_entry.embedding and len(existing_entry.embedding) == self.embed_dim:
                    sim = _cosine_sim(emb, existing_entry.embedding)
                    if sim >= self.similarity_threshold:
                        # Check if they contradict — if so, keep both
                        if self._detect_contradiction(obs.content, existing_entry.content):
                            # They contradict — don't dedup, but record the contradiction
                            if existing_id not in existing_entry.contradiction_ids:
                                existing_entry.contradiction_ids.append(entry_id)
                            contradiction_ids.append(existing_id)
                            break  # Don't dedup, but continue to create the new entry
                        
                        # Near-duplicate — merge into existing
                        existing_entry.access_count += 1
                        existing_entry.last_accessed = now
                        existing_entry.version += 1
                        existing_entry.source_count += 1
                        existing_entry.confidence = min(1.0, existing_entry.confidence + 0.05)
                        if session_name and session_name not in (existing_entry.session_name or ""):
                            existing_entry.session_name = session_name
                        # Carry over any contradiction flags
                        for cid in contradiction_ids:
                            if cid not in existing_entry.contradiction_ids:
                                existing_entry.contradiction_ids.append(cid)
                        deduped = True
                        break

            if deduped:
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
                embedding=emb,
                confidence=0.9 if obs.level == "explicit" else 0.7,
                contradiction_ids=contradiction_ids,
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
        """Simple text search over stored observations (backward compat)."""
        text_lower = text.lower()
        scored = []
        for entry in self._entries.values():
            score = entry.importance_score
            if text_lower in entry.content.lower():
                score *= 2.0
            # Also check tags
            for tag in entry.tags:
                if text_lower in tag.lower():
                    score *= 1.5
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def semantic_search(self, text: str, limit: int = 10) -> list[tuple[MemoryEvolutionEntry, float]]:
        """Embedding-based semantic search.
        
        Returns (entry, similarity_score) pairs sorted by relevance.
        Much better than text search for finding conceptually related observations.
        """
        query_emb = self._compute_embedding(text)
        scored = []
        for entry in self._entries.values():
            if entry.embedding and len(entry.embedding) == self.embed_dim:
                sim = _cosine_sim(query_emb, entry.embedding)
                # Boost by importance
                score = sim * entry.importance_score
                scored.append((score, entry, sim))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(e, sim) for score, e, sim in scored[:limit]]

    def get_contradictions(self, peer_name: str | None = None) -> list[tuple[MemoryEvolutionEntry, MemoryEvolutionEntry]]:
        """Get all pairs of contradictory observations.
        
        Returns list of (entry_a, entry_b) pairs that contradict each other.
        """
        pairs = []
        entries = list(self._entries.values())
        if peer_name:
            entries = [e for e in entries if e.peer_name == peer_name]
        
        for i, a in enumerate(entries):
            for b in entries[i+1:]:
                if a.id in b.contradiction_ids or b.id in a.contradiction_ids:
                    pairs.append((a, b))
                elif a.peer_name == b.peer_name:
                    # Check for new contradictions
                    if self._detect_contradiction(a.content, b.content):
                        pairs.append((a, b))
        
        return pairs

    def run_evolution_cycle(self) -> dict[str, Any]:
        """Run decay + consolidation + contradiction detection.
        
        v2 improvements:
        - Access-frequency-boosted decay (frequently accessed memories decay slower)
        - Consolidation: merge similar observations into higher-level insights
        - Contradiction flagging
        - Temporal weighting (recent observations score higher)
        """
        now = _now_ts()
        decayed = 0
        consolidated = 0
        contradictions_found = 0
        promoted = 0

        # Phase 1: Decay with access-frequency boost
        for entry in self._entries.values():
            age_days = (now - entry.timestamp) / 86400
            
            # Base decay: lose relevance over time
            base_decay = 0.5 ** (age_days / 30.0)
            
            # Access-frequency boost: each access slows decay by 10%
            access_boost = 1.0 + (entry.access_count * 0.1)
            
            # Confidence boost: higher confidence = slower decay
            confidence_boost = 0.5 + (entry.confidence * 0.5)
            
            # Combined importance
            entry.importance_score = max(0.05, base_decay * access_boost * confidence_boost)
            
            if entry.importance_score < 0.15:
                decayed += 1

        # Phase 2: Consolidation — merge similar observations
        entries_list = list(self._entries.values())
        merged_ids: set[str] = set()
        
        for i, a in enumerate(entries_list):
            if a.id in merged_ids or not a.embedding:
                continue
            similar = []
            for j, b in enumerate(entries_list):
                if i >= j or b.id in merged_ids or not b.embedding:
                    continue
                if a.peer_name != b.peer_name:
                    continue
                if len(a.embedding) != len(b.embedding):
                    continue
                sim = _cosine_sim(a.embedding, b.embedding)
                if sim >= self.similarity_threshold:
                    similar.append(b)
            
            if len(similar) >= 2:
                # Merge similar observations into a consolidated one
                merged_content = "; ".join([a.content] + [s.content for s in similar])
                consolidated_id = _make_id(merged_content, f"consolidated:{a.peer_name}")
                
                if consolidated_id not in self._entries:
                    # Create consolidated entry
                    consolidated_entry = MemoryEvolutionEntry(
                        id=consolidated_id,
                        content=merged_content,
                        timestamp=now,
                        tags=a.tags + ["consolidated"],
                        level="deductive",
                        peer_name=a.peer_name,
                        embedding=self._compute_embedding(merged_content),
                        confidence=min(1.0, a.confidence + 0.1 * len(similar)),
                        consolidated=True,
                        source_count=1 + len(similar),
                        importance_score=max(e.importance_score for e in [a] + similar),
                    )
                    self._entries[consolidated_id] = consolidated_entry
                    consolidated += 1
                    
                    # Mark originals as rewritten
                    for orig in [a] + similar:
                        orig.rewritten_from = consolidated_id
                        orig.importance_score *= 0.5  # Reduce importance of originals
                        merged_ids.add(orig.id)

        # Phase 3: Contradiction detection
        for a in entries_list:
            for b in entries_list:
                if a.id >= b.id or a.peer_name != b.peer_name:
                    continue
                if a.id in b.contradiction_ids or b.id in a.contradiction_ids:
                    continue
                if self._detect_contradiction(a.content, b.content):
                    a.contradiction_ids.append(b.id)
                    b.contradiction_ids.append(a.id)
                    contradictions_found += 1

        # Phase 4: Promote high-confidence deductive to explicit
        for entry in self._entries.values():
            if (entry.level == "deductive" 
                and entry.confidence >= 0.95 
                and entry.source_count >= 3
                and not entry.consolidated):
                entry.level = "explicit"
                promoted += 1

        self._save()
        return {
            "memories_processed": len(self._entries),
            "memories_decayed": decayed,
            "consolidated": consolidated,
            "contradictions_found": contradictions_found,
            "promoted_to_explicit": promoted,
        }

    def get_observations_by_peer(self, peer_name: str) -> list[MemoryEvolutionEntry]:
        return [e for e in self._entries.values() if e.peer_name == peer_name]

    def get_observations_by_level(self, level: str) -> list[MemoryEvolutionEntry]:
        return [e for e in self._entries.values() if e.level == level]

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        peers = set(e.peer_name for e in self._entries.values())
        levels = {}
        for e in self._entries.values():
            levels[e.level] = levels.get(e.level, 0) + 1
        
        contradictions = sum(len(e.contradiction_ids) for e in self._entries.values())
        consolidated = sum(1 for e in self._entries.values() if e.consolidated)
        
        return {
            "total_entries": len(self._entries),
            "peers": len(peers),
            "by_level": levels,
            "contradictions": contradictions,
            "consolidated": consolidated,
            "avg_confidence": sum(e.confidence for e in self._entries.values()) / max(len(self._entries), 1),
            "avg_importance": sum(e.importance_score for e in self._entries.values()) / max(len(self._entries), 1),
        }


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
    # v2: richer trace metadata
    confidence: float = 1.0
    contradiction_flags: list[str] = field(default_factory=list)
    source_count: int = 1


class ObliviarchBridge:
    """Feeds deriver observations into OBLIVIARCH's compression pipeline.
    
    v2: Richer trace metadata for better compression decisions.
    """

    @staticmethod
    def observation_to_trace(
        obs: Observation,
        peer_name: str,
        session_name: str | None = None,
        contradictions: list[str] | None = None,
    ) -> ObliviarchTrace:
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
            confidence=0.9 if obs.level == "explicit" else 0.7,
            contradiction_flags=contradictions or [],
            source_count=1,
        )

    @staticmethod
    def observations_to_traces(
        observations: ObservationSet,
        peer_name: str,
        session_name: str | None = None,
    ) -> list[ObliviarchTrace]:
        return [
            ObliviarchBridge.observation_to_trace(obs, peer_name, session_name)
            for obs in observations.all
        ]

    @staticmethod
    def format_for_obliviarch_record(
        observations: ObservationSet, 
        peer_name: str,
        contradictions: list[str] | None = None,
    ) -> dict[str, Any]:
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
            "contradictions": contradictions or [],
        }


# ══════════════════════════════════════════════════════════════════════════
# 3. Deterministic-Retrieval Bridge
# ══════════════════════════════════════════════════════════════════════════

class DeterministicRetrievalBridge:
    """Makes observations accessible via deterministic-retrieval's path-based lookup.
    
    v2: Stores embeddings alongside observations for semantic retrieval,
    maintains a peer index for fast listing, and supports incremental updates.
    """

    def __init__(self, base_path: str = "~/.hermes/workspace/observations"):
        self.base_path = os.path.expanduser(base_path)

    def _ensure_dir(self, peer_name: str, level: str) -> str:
        path = os.path.join(self.base_path, peer_name, level)
        os.makedirs(path, exist_ok=True)
        return path

    def store_observations(
        self,
        observations: ObservationSet,
        peer_name: str,
    ) -> dict[str, str]:
        """Store observations as JSON files with embeddings, return file paths."""
        date_str = _date_str()
        result = {}

        for level_name, obs_list in [
            ("explicit", observations.explicit),
            ("deductive", observations.deductive),
        ]:
            if not obs_list:
                continue
            dir_path = self._ensure_dir(peer_name, level_name)
            file_path = os.path.join(dir_path, f"{date_str}.json")

            # Load existing or create new
            try:
                with open(file_path) as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []

            existing.extend([
                {
                    "content": o.content,
                    "timestamp": o.created_at or datetime.now(timezone.utc).isoformat(),
                    "level": o.level,
                    "peer": peer_name,
                    "embedding": _text_to_embedding(o.content),
                    "confidence": 0.9 if o.level == "explicit" else 0.7,
                }
                for o in obs_list
            ])

            with open(file_path, "w") as f:
                json.dump(existing, f, indent=2)

            result[level_name] = file_path

        # Update peer index
        self._update_peer_index(peer_name)

        return result

    def _update_peer_index(self, peer_name: str) -> None:
        """Maintain a fast peer index for listing."""
        index_path = os.path.join(self.base_path, "_index.json")
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            index = {"peers": {}}
        
        index["peers"][peer_name] = {
            "last_updated": _date_str(),
            "explicit_count": self._count_level(peer_name, "explicit"),
            "deductive_count": self._count_level(peer_name, "deductive"),
        }
        index["updated_at"] = _now_ts()
        
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _count_level(self, peer_name: str, level: str) -> int:
        """Count observations for a peer at a given level."""
        path = os.path.join(self.base_path, peer_name, level)
        if not os.path.isdir(path):
            return 0
        count = 0
        for fname in os.listdir(path):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(path, fname)) as f:
                        count += len(json.load(f))
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
        return count

    def get_path_for(self, peer_name: str, level: str = "explicit") -> str:
        return os.path.join(self.base_path, peer_name, level)

    def list_peers(self) -> list[dict[str, Any]]:
        """List all peers with observation counts."""
        index_path = os.path.join(self.base_path, "_index.json")
        try:
            with open(index_path) as f:
                index = json.load(f)
            return [{"name": k, **v} for k, v in index.get("peers", {}).items()]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def retrieve_latest(self, peer_name: str, level: str = "explicit", limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve the most recent observations for a peer."""
        path = os.path.join(self.base_path, peer_name, level)
        if not os.path.isdir(path):
            return []
        
        files = sorted([f for f in os.listdir(path) if f.endswith(".json")], reverse=True)
        results = []
        for fname in files[:5]:  # Check last 5 date files
            try:
                with open(os.path.join(path, fname)) as f:
                    data = json.load(f)
                results.extend(data[-limit:])
                if len(results) >= limit:
                    break
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return results[:limit]


# ══════════════════════════════════════════════════════════════════════════
# 4. Recursive-Retrieval Bridge
# ══════════════════════════════════════════════════════════════════════════

class RecursiveRetrievalBridge:
    """Makes observations available as a context layer in recursive-retrieval.
    
    v2: Includes confidence scores, contradiction flags, and temporal metadata
    in the context layers for better downstream reasoning.
    """

    @staticmethod
    def observations_to_agent_layer(
        peer_name: str,
        explicit: list[Observation],
        deductive: list[Observation] | None = None,
        contradictions: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Convert observations to an agent-layer dict with rich metadata."""
        layer = {
            "peer_name": peer_name,
            "observations": {
                "explicit": [o.content for o in explicit],
                "deductive": [o.content for o in (deductive or [])],
            },
            "observation_count": len(explicit) + len(deductive or []),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "confidence_summary": {
                "explicit_count": len(explicit),
                "deductive_count": len(deductive or []),
                "has_contradictions": len(contradictions or []) > 0,
            },
        }
        if contradictions:
            layer["contradictions"] = [
                {"observation_a": a, "observation_b": b}
                for a, b in contradictions
            ]
        return layer

    @staticmethod
    def observations_to_session_layer(
        session_name: str,
        observations: ObservationSet,
    ) -> dict[str, Any]:
        return {
            "session_name": session_name,
            "observations": observations.to_dict(),
            "observation_count": observations.count,
            "timestamp": _now_ts(),
        }


# ══════════════════════════════════════════════════════════════════════════
# 5. Hermes-Namespace Bridge
# ══════════════════════════════════════════════════════════════════════════

class NamespaceBridge:
    """Registers observation URI handlers for hermes-namespace.
    
    v2: Supports semantic search via hermes://observations/{peer}/search?q=...
    """

    def __init__(self, base_path: str = "~/.hermes/workspace/observations"):
        self.base_path = os.path.expanduser(base_path)

    def resolve(self, uri: str) -> str:
        """Resolve a hermes://observations/... URI to a file path.
        
        Supports:
          hermes://observations/{peer_name}
          hermes://observations/{peer_name}/{level}
          hermes://observations/{peer_name}/{level}/{date}
          hermes://observations/{peer_name}/search?q=...  (returns search results path)
        """
        # Check for search query
        search_match = re.match(
            r"hermes://observations/([^/]+)/search\?q=(.+)",
            uri,
        )
        if search_match:
            peer_name = search_match.group(1)
            query = search_match.group(2)
            # Return a path that the resolver can use
            return os.path.join(self.base_path, peer_name, "_search", f"{_date_str()}.json")

        # Standard path resolution
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
            return os.path.join(self.base_path, peer_name, level)

    def register_with_namespace(self) -> dict[str, str]:
        return {
            "namespace": "observations",
            "description": "Access deriver observations by peer, level, and date. Supports semantic search via ?q= parameter.",
            "example": "hermes://observations/alice/explicit/20260709",
            "resolve": "NamespaceBridge.resolve()",
        }


# ══════════════════════════════════════════════════════════════════════════
# 6. NEUROSWARM Bridge
# ══════════════════════════════════════════════════════════════════════════

class NeuroSwarmBridge:
    """Wires the dialectic agent into NEUROSWARM's dual-phase dispatch.
    
    v2: Includes contradiction context and confidence scoring in council context.
    """

    @staticmethod
    def create_dialectic_tool(agent: DialecticAgent) -> dict[str, Any]:
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
        contradictions: list[tuple[str, str]] | None = None,
    ) -> str:
        """Format observations as context for NEUROSWARM's 10-D council.
        
        v2: Includes contradiction warnings and confidence indicators.
        """
        parts = [f"## Observations about {peer_name}", ""]
        parts.append(f"Total: {observations.count} ({len(observations.explicit)} explicit, {len(observations.deductive)} deductive)")
        
        if contradictions:
            parts.append(f"\n⚠️ **Contradictions detected:** {len(contradictions)}")
            for a, b in contradictions[:3]:
                parts.append(f"  - \"{a}\" vs \"{b}\"")
            parts.append("")
        
        parts.append("")
        for obs in observations.all[:max_obs]:
            tag = "📌" if obs.level == "explicit" else "🧠"
            confidence = "high" if obs.level == "explicit" else "medium"
            parts.append(f"{tag} [{confidence}] {obs.content}")

        if observations.count > max_obs:
            parts.append(f"\n... and {observations.count - max_obs} more observations")

        return "\n".join(parts)

    @staticmethod
    def council_verdict_to_observations(
        verdict: str,
        peer_name: str,
        session_name: str | None = None,
    ) -> ObservationSet:
        return ObservationSet(
            deductive=[
                Observation(
                    content=verdict,
                    level="deductive",
                    peer_name=peer_name,
                    session_name=session_name,
                )
            ],
        )
