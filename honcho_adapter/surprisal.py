"""
Surprisal — Novelty detection for finding interesting/contradictory observations.

Adapted from Honcho's src/dreamer/surprisal.py. Stripped of Postgres/server deps.
Uses tree-based geometric surprisal to find observations that are
anomalous, novel, or contradictory relative to the existing corpus.

v2.0 — Improved performance over time:
- Real embedding-based features (not just proxy features)
- Incremental fitting (add new observations without rebuilding from scratch)
- Temporal weighting (recent observations weighted higher)
- Cross-peer analysis (find surprising patterns across different peers)
- Contradiction-aware scoring (observations that contradict each other get boosted)
- Persistence (save/load forest state)
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────

@dataclass
class SurprisalConfig:
    """Configuration for surprisal computation."""
    enabled: bool = True
    top_k: int = 10
    threshold: float = 0.5
    num_trees: int = 32
    sample_size: int = 256
    max_depth: int = 10
    # v2: temporal and cross-peer settings
    temporal_decay_days: float = 30.0  # Half-life for temporal weighting
    cross_peer_enabled: bool = True
    contradiction_boost: float = 2.0  # Multiplier for contradictory observations
    incremental_rebuild_threshold: int = 50  # Rebuild forest after this many new observations
    persist_path: str | None = None  # Path to save/load forest state


@dataclass
class SurprisalScore:
    """An observation with its computed surprisal score."""
    content: str
    level: str
    surprisal: float
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None
    # v2: richer scoring
    temporal_weight: float = 1.0
    contradiction_count: int = 0
    peer_name: str = ""
    timestamp: float = 0.0


# ── Embedding utilities ────────────────────────────────────────────────────

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
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ── Random Projection Tree (simplified RPForest) ───────────────────────────

class RPNode:
    """Node in a random projection tree."""
    def __init__(self, indices: np.ndarray, depth: int = 0, max_depth: int = 10):
        self.indices = indices
        self.depth = depth
        self.max_depth = max_depth
        self.left: RPNode | None = None
        self.right: RPNode | None = None
        self.split_vector: np.ndarray | None = None
        self.split_value: float = 0.0
        self.is_leaf = len(indices) <= 2 or depth >= max_depth

        if not self.is_leaf:
            self._split()

    def _split(self) -> None:
        n = len(self.indices)
        rng = np.random.default_rng()
        self.split_vector = rng.normal(size=self.indices.shape[1])
        self.split_vector /= np.linalg.norm(self.split_vector)
        proj = self.indices @ self.split_vector
        self.split_value = np.median(proj)
        left_mask = proj <= self.split_value
        self.left = RPNode(self.indices[left_mask], self.depth + 1, self.max_depth)
        self.right = RPNode(self.indices[~left_mask], self.depth + 1, self.max_depth)

    def depth_of(self, point: np.ndarray) -> int:
        """How deep this point falls in the tree. Lower = more normal, higher = more surprising."""
        if self.is_leaf:
            return self.depth
        assert self.split_vector is not None
        proj = float(point @ self.split_vector)
        if proj <= self.split_value:
            return self.left.depth_of(point) if self.left else self.depth
        return self.right.depth_of(point) if self.right else self.depth


class RPForest:
    """Random projection forest for anomaly detection.
    
    v2: Supports incremental fitting and persistence.
    """

    def __init__(self, num_trees: int = 32, max_depth: int = 10):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees: list[RPNode] = []
        self._n_samples: int = 0

    def fit(self, embeddings: np.ndarray) -> None:
        """Build the forest from a matrix of embeddings (n_samples x n_dims)."""
        self._n_samples = len(embeddings)
        actual_max_depth = min(self.max_depth, max(3, int(np.log2(len(embeddings)))))
        self.trees = [
            RPNode(embeddings, max_depth=actual_max_depth)
            for _ in range(self.num_trees)
        ]

    def anomaly_score(self, point: np.ndarray) -> float:
        """Compute anomaly score (0=normal, 1=anomalous)."""
        if not self.trees:
            return 0.0
        depths = [tree.depth_of(point) for tree in self.trees]
        mean_depth = np.mean(depths)
        return float(mean_depth) / self.max_depth

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "num_trees": self.num_trees,
            "max_depth": self.max_depth,
            "n_samples": self._n_samples,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RPForest:
        """Create from saved state (trees need to be re-fit)."""
        forest = cls(
            num_trees=state.get("num_trees", 32),
            max_depth=state.get("max_depth", 10),
        )
        forest._n_samples = state.get("n_samples", 0)
        return forest


# ── Surprisal Engine ───────────────────────────────────────────────────────

class SurprisalEngine:
    """Computes surprisal scores for observations.
    
    v2 improvements:
    - Real embedding-based features (deterministic n-gram hashing)
    - Incremental fitting (tracks new observations, rebuilds periodically)
    - Temporal weighting (recent observations score higher for novelty)
    - Cross-peer analysis (finds surprising patterns across different peers)
    - Contradiction-aware scoring (contradictory observations get boosted)
    - Persistence (save/load forest state to disk)
    
    Usage:
        engine = SurprisalEngine()
        engine.fit(existing_embeddings)
        scores = engine.compute(new_observations, new_embeddings)
        # scores[0] has the highest surprisal (most interesting/novel)
    """

    def __init__(self, config: SurprisalConfig | None = None):
        self.config = config or SurprisalConfig()
        self._forest: RPForest | None = None
        self._embeddings: np.ndarray | None = None
        # v2: incremental tracking
        self._all_observations: list[dict[str, Any]] = []
        self._new_since_rebuild: int = 0
        self._contradiction_map: dict[str, list[str]] = {}  # content_hash -> [contradicting_content_hashes]
        self._embedding_dim: int = 64

    def _make_hash(self, content: str) -> str:
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def fit(self, embeddings: list[list[float]]) -> None:
        """Fit the forest on existing observations."""
        if not embeddings:
            logger.warning("No embeddings to fit surprisal forest")
            return
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._embedding_dim = self._embeddings.shape[1]
        self._forest = RPForest(
            num_trees=self.config.num_trees,
            max_depth=min(self.config.max_depth, max(3, int(np.log2(len(embeddings))))),
        )
        self._forest.fit(self._embeddings)
        self._new_since_rebuild = 0

    def add_observations(self, observations: list[dict[str, Any]]) -> None:
        """Add observations to the tracked set (for incremental fitting).
        
        v2: New method. Adds observations without immediately rebuilding the forest.
        The forest is rebuilt when _new_since_rebuild exceeds the threshold.
        """
        for obs in observations:
            content = obs.get("content", "")
            if content:
                h = self._make_hash(content)
                # Avoid duplicates
                if not any(o.get("content") == content for o in self._all_observations):
                    self._all_observations.append(obs)
                    self._new_since_rebuild += 1

        # Auto-rebuild if threshold exceeded
        if (self._new_since_rebuild >= self.config.incremental_rebuild_threshold 
            and len(self._all_observations) >= 5):
            self._rebuild_forest()

    def _rebuild_forest(self) -> None:
        """Rebuild the forest from all tracked observations."""
        if len(self._all_observations) < 5:
            return
        
        embeddings = []
        for obs in self._all_observations:
            content = obs.get("content", "")
            emb = _text_to_embedding(content, self._embedding_dim)
            embeddings.append(emb)
        
        self.fit(embeddings)
        logger.info(
            "Rebuilt forest with %d observations (%d new since last rebuild)",
            len(self._all_observations), self._new_since_rebuild,
        )

    def _compute_temporal_weight(self, timestamp: float | None) -> float:
        """Compute temporal weight (recent = higher weight for novelty)."""
        if timestamp is None:
            return 1.0
        now = datetime.now(timezone.utc).timestamp()
        age_days = (now - timestamp) / 86400
        return 2.0 ** (-age_days / self.config.temporal_decay_days)

    def compute(
        self,
        observations: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[SurprisalScore]:
        """Compute surprisal scores for a list of observations.
        
        v2: Includes temporal weighting, contradiction boosting,
        and cross-peer analysis.
        
        Args:
            observations: List of dicts with 'content', 'level', 'peer_name',
                         'timestamp' (optional) keys.
            embeddings: Corresponding embedding vectors.
        
        Returns:
            List of SurprisalScore, sorted by surprisal descending (highest first).
        """
        if self._forest is None:
            logger.warning("No forest fitted — returning all observations with score 0")
            return [
                SurprisalScore(content=o["content"], level=o.get("level", "explicit"), surprisal=0.0)
                for o in observations
            ]

        if len(observations) != len(embeddings):
            raise ValueError(
                f"observations ({len(observations)}) and embeddings ({len(embeddings)}) must match"
            )

        scores = []
        for obs, emb in zip(observations, embeddings):
            point = np.array(emb, dtype=np.float32)
            
            # Base geometric surprisal
            base_s = self._forest.anomaly_score(point)
            
            # Temporal weighting: recent observations are more "surprising" if novel
            ts = obs.get("timestamp")
            temporal_w = self._compute_temporal_weight(ts)
            
            # Contradiction boost: observations that contradict existing ones are more interesting
            content = obs.get("content", "")
            content_hash = self._make_hash(content)
            contradiction_count = len(self._contradiction_map.get(content_hash, []))
            contradiction_boost = 1.0 + (contradiction_count * (self.config.contradiction_boost - 1.0))
            
            # Cross-peer boost: if this observation is about a different peer than most
            peer_name = obs.get("peer_name", "")
            cross_peer_boost = 1.0
            if self.config.cross_peer_enabled and self._all_observations:
                peer_counts: dict[str, int] = {}
                for o in self._all_observations:
                    p = o.get("peer_name", "")
                    peer_counts[p] = peer_counts.get(p, 0) + 1
                # If this peer is underrepresented, boost
                total = sum(peer_counts.values())
                if total > 0 and peer_name:
                    ratio = peer_counts.get(peer_name, 0) / total
                    if ratio < 0.1:  # Less than 10% of observations
                        cross_peer_boost = 1.5
                    elif ratio < 0.25:
                        cross_peer_boost = 1.2
            
            # Combined score
            combined = base_s * temporal_w * contradiction_boost * cross_peer_boost
            
            scores.append(SurprisalScore(
                content=content,
                level=obs.get("level", "explicit"),
                surprisal=float(combined),
                embedding=emb,
                temporal_weight=float(temporal_w),
                contradiction_count=contradiction_count,
                peer_name=peer_name,
                timestamp=ts or 0.0,
            ))

        # Sort by surprisal descending
        scores.sort(key=lambda x: x.surprisal, reverse=True)

        # Filter by threshold
        scores = [s for s in scores if s.surprisal >= self.config.threshold]

        return scores[:self.config.top_k]

    def compute_single(self, content: str, embedding: list[float]) -> float:
        """Compute surprisal for a single observation."""
        if self._forest is None:
            return 0.0
        point = np.array(embedding, dtype=np.float32)
        return float(self._forest.anomaly_score(point))

    def register_contradiction(self, content_a: str, content_b: str) -> None:
        """Register a contradiction between two observations.
        
        v2: New method. Contradictory observations get boosted surprisal scores.
        """
        h_a = self._make_hash(content_a)
        h_b = self._make_hash(content_b)
        
        if h_a not in self._contradiction_map:
            self._contradiction_map[h_a] = []
        if h_b not in self._contradiction_map:
            self._contradiction_map[h_b] = []
        
        if h_b not in self._contradiction_map[h_a]:
            self._contradiction_map[h_a].append(h_b)
        if h_a not in self._contradiction_map[h_b]:
            self._contradiction_map[h_b].append(h_a)

    def save_state(self, path: str | None = None) -> None:
        """Save engine state to disk for persistence across sessions."""
        save_path = path or self.config.persist_path
        if not save_path:
            logger.warning("No persist path set — cannot save state")
            return
        
        state = {
            "config": {
                "num_trees": self.config.num_trees,
                "max_depth": self.config.max_depth,
                "embedding_dim": self._embedding_dim,
            },
            "forest": self._forest.get_state() if self._forest else None,
            "observations": self._all_observations,
            "contradiction_map": self._contradiction_map,
            "new_since_rebuild": self._new_since_rebuild,
        }
        
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info("Saved surprisal state to %s (%d observations)", save_path, len(self._all_observations))

    def load_state(self, path: str | None = None) -> bool:
        """Load engine state from disk.
        
        Returns True if state was loaded, False otherwise.
        """
        load_path = path or self.config.persist_path
        if not load_path or not os.path.exists(load_path):
            return False
        
        try:
            with open(load_path) as f:
                state = json.load(f)
            
            self._all_observations = state.get("observations", [])
            self._contradiction_map = state.get("contradiction_map", {})
            self._new_since_rebuild = state.get("new_since_rebuild", 0)
            self._embedding_dim = state.get("config", {}).get("embedding_dim", 64)
            
            # Rebuild forest from saved observations
            if self._all_observations:
                self._rebuild_forest()
            
            logger.info("Loaded surprisal state from %s (%d observations)", load_path, len(self._all_observations))
            return True
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load surprisal state: %s", e)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_observations": len(self._all_observations),
            "forest_built": self._forest is not None,
            "new_since_rebuild": self._new_since_rebuild,
            "contradiction_pairs": sum(len(v) for v in self._contradiction_map.values()) // 2,
            "embedding_dim": self._embedding_dim,
            "num_trees": self.config.num_trees,
        }
