"""
Surprisal — Novelty detection for finding interesting/contradictory observations.

Adapted from Honcho's src/dreamer/surprisal.py. Stripped of Postgres/server deps.
Uses tree-based geometric surprisal to find observations that are
anomalous, novel, or contradictory relative to the existing corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class SurprisalScore:
    """An observation with its computed surprisal score."""
    content: str
    level: str
    surprisal: float
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


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
        # Random projection
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
    """Random projection forest for anomaly detection."""

    def __init__(self, num_trees: int = 32, max_depth: int = 10):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees: list[RPNode] = []

    def fit(self, embeddings: np.ndarray) -> None:
        """Build the forest from a matrix of embeddings (n_samples x n_dims)."""
        self.trees = [
            RPNode(embeddings, max_depth=self.max_depth)
            for _ in range(self.num_trees)
        ]

    def anomaly_score(self, point: np.ndarray) -> float:
        """Compute anomaly score (0=normal, 1=anomalous)."""
        if not self.trees:
            return 0.0
        depths = [tree.depth_of(point) for tree in self.trees]
        mean_depth = np.mean(depths)
        # Normalize: deeper = more anomalous
        max_depth = self.max_depth
        return mean_depth / max_depth


# ── Surprisal Engine ───────────────────────────────────────────────────────

class SurprisalEngine:
    """Computes surprisal scores for observations.

    Usage:
        engine = SurprisalEngine()
        engine.fit(existing_embeddings)  # from previously stored observations
        scores = engine.compute(new_observations)
        # scores[0] has the highest surprisal (most interesting/novel)
    """

    def __init__(self, config: SurprisalConfig | None = None):
        self.config = config or SurprisalConfig()
        self._forest: RPForest | None = None
        self._embeddings: np.ndarray | None = None

    def fit(self, embeddings: list[list[float]]) -> None:
        """Fit the forest on existing observations."""
        if not embeddings:
            logger.warning("No embeddings to fit surprisal forest")
            return
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._forest = RPForest(
            num_trees=self.config.num_trees,
            max_depth=min(10, max(3, int(np.log2(len(embeddings))))),
        )
        self._forest.fit(self._embeddings)

    def compute(
        self,
        observations: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[SurprisalScore]:
        """Compute surprisal scores for a list of observations.

        Args:
            observations: List of dicts with 'content' and 'level' keys.
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
            s = self._forest.anomaly_score(point)
            scores.append(SurprisalScore(
                content=obs["content"],
                level=obs.get("level", "explicit"),
                surprisal=float(s),
                embedding=emb,
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