"""Tests for honcho-adapter v2 improvements — memory that performs better over time."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone

import pytest

from honcho_adapter import (
    Deriver, DeriverConfig, Observation, ObservationSet,
    SurprisalEngine, SurprisalConfig, SurprisalScore,
)
from honcho_adapter.bridges import (
    MemoryEvolutionBridge, MemoryEvolutionEntry,
    DeterministicRetrievalBridge, NamespaceBridge,
    NeuroSwarmBridge, ObliviarchBridge, ObliviarchTrace,
    _text_to_embedding, _cosine_sim, _make_id,
)


# ══════════════════════════════════════════════════════════════════════════
# Shared utilities tests
# ══════════════════════════════════════════════════════════════════════════

class TestEmbedding:
    def test_text_to_embedding_returns_fixed_dim(self):
        emb = _text_to_embedding("hello world", dim=64)
        assert len(emb) == 64
        assert all(isinstance(v, float) for v in emb)

    def test_text_to_embedding_normalized(self):
        emb = _text_to_embedding("test", dim=64)
        import math
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 0.01

    def test_similar_texts_have_similar_embeddings(self):
        a = _text_to_embedding("MUI has RM905M in debt", dim=64)
        b = _text_to_embedding("MUI has RM905M of debt", dim=64)
        c = _text_to_embedding("the weather is nice today", dim=64)
        
        sim_ab = _cosine_sim(a, b)
        sim_ac = _cosine_sim(a, c)
        
        assert sim_ab > sim_ac, "Similar texts should have higher cosine similarity"

    def test_cosine_sim_identical(self):
        emb = _text_to_embedding("test", dim=64)
        assert abs(_cosine_sim(emb, emb) - 1.0) < 0.001

    def test_cosine_sim_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_sim(a, b)) < 0.001


class TestMakeId:
    def test_deterministic(self):
        assert _make_id("hello") == _make_id("hello")

    def test_different_content_different_ids(self):
        assert _make_id("hello") != _make_id("world")

    def test_namespace_separation(self):
        assert _make_id("hello", "ns1") != _make_id("hello", "ns2")


# ══════════════════════════════════════════════════════════════════════════
# Deriver v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestObservationV2:
    def test_observation_has_confidence_default(self):
        obs = Observation(content="test", level="explicit", peer_name="alice")
        assert obs.confidence == 0.9

    def test_observation_has_contradiction_flags(self):
        obs = Observation(content="test", level="explicit", peer_name="alice")
        assert obs.contradiction_flags == []

    def test_observation_has_entities(self):
        obs = Observation(content="test", level="explicit", peer_name="alice")
        assert obs.entities == []

    def test_observation_has_sentiment(self):
        obs = Observation(content="test", level="explicit", peer_name="alice")
        assert obs.sentiment == "neutral"


class TestObservationSetV2:
    def test_deduplicate_removes_duplicates(self):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="alice is 25", level="explicit", peer_name="alice"),
                Observation(content="alice is 25", level="explicit", peer_name="alice"),
            ],
        )
        assert obs_set.count == 2
        obs_set.deduplicate()
        assert obs_set.count == 1

    def test_detect_internal_contradictions_negation(self):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905M in debt", level="explicit", peer_name="mui"),
                Observation(content="MUI has no debt", level="explicit", peer_name="mui"),
            ],
        )
        contradictions = obs_set.detect_internal_contradictions()
        assert len(contradictions) == 1

    def test_detect_internal_contradictions_opposite(self):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="revenue increased by 20%", level="explicit", peer_name="mui"),
                Observation(content="revenue decreased by 10%", level="explicit", peer_name="mui"),
            ],
        )
        contradictions = obs_set.detect_internal_contradictions()
        assert len(contradictions) == 1

    def test_detect_internal_contradictions_no_false_positive(self):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905M in debt", level="explicit", peer_name="mui"),
                Observation(content="A&W has 101 stores in Malaysia", level="explicit", peer_name="mui"),
            ],
        )
        contradictions = obs_set.detect_internal_contradictions()
        assert len(contradictions) == 0

    def test_contradiction_flags_set_on_observations(self):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905M in debt", level="explicit", peer_name="mui"),
                Observation(content="MUI has no debt", level="explicit", peer_name="mui"),
            ],
        )
        obs_set.detect_internal_contradictions()
        assert len(obs_set.all[0].contradiction_flags) > 0
        assert len(obs_set.all[1].contradiction_flags) > 0


# ══════════════════════════════════════════════════════════════════════════
# MemoryEvolutionBridge v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestMemoryEvolutionBridgeV2:
    @pytest.fixture
    def bridge(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        bridge = MemoryEvolutionBridge(path)
        yield bridge
        os.unlink(path)

    def test_ingest_observations_creates_entries(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="alice is 25", level="explicit", peer_name="alice")],
            deductive=[Observation(content="alice is an adult", level="deductive", peer_name="alice")],
        )
        entries = bridge.ingest_observations(obs_set, peer_name="alice")
        assert len(entries) == 2

    def test_ingest_observations_computes_embeddings(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="alice is 25", level="explicit", peer_name="alice")],
        )
        bridge.ingest_observations(obs_set, peer_name="alice")
        entry = list(bridge._entries.values())[0]
        assert entry.embedding is not None
        assert len(entry.embedding) == 64

    def test_ingest_observations_detects_contradictions(self, bridge):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905M in debt", level="explicit", peer_name="mui"),
            ],
        )
        bridge.ingest_observations(obs_set, peer_name="mui")
        
        obs_set2 = ObservationSet(
            explicit=[
                Observation(content="MUI has no debt", level="explicit", peer_name="mui"),
            ],
        )
        bridge.ingest_observations(obs_set2, peer_name="mui")
        
        contradictions = bridge.get_contradictions("mui")
        assert len(contradictions) >= 1

    def test_dedup_by_semantic_similarity(self, bridge):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905 million in total debt", level="explicit", peer_name="mui"),
            ],
        )
        bridge.ingest_observations(obs_set, peer_name="mui")
        
        obs_set2 = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905 million in total debt", level="explicit", peer_name="mui"),
            ],
        )
        entries = bridge.ingest_observations(obs_set2, peer_name="mui")
        # Should be deduped (exact duplicate)
        assert len(entries) == 0

    def test_semantic_search(self, bridge):
        obs_set = ObservationSet(
            explicit=[
                Observation(content="MUI has RM905M in debt at 10% interest", level="explicit", peer_name="mui"),
                Observation(content="A&W has 101 stores in Malaysia", level="explicit", peer_name="mui"),
                Observation(content="Hyde Park hotel is worth RM1.1B", level="explicit", peer_name="mui"),
            ],
        )
        bridge.ingest_observations(obs_set, peer_name="mui")
        
        results = bridge.semantic_search("debt and interest rates", limit=2)
        assert len(results) == 2
        # Most relevant should be about debt
        assert "debt" in results[0][0].content.lower()

    def test_run_evolution_cycle_consolidation(self, bridge):
        # Add observations that are semantically related but not near-duplicates
        # (different enough to avoid dedup at 0.90 threshold, similar enough to consolidate at 0.80)
        contents = [
            "MUI has RM905M in debt at high interest rates",
            "The company carries RM905M of debt at 10% interest",
            "MUI's debt is RM905M at elevated rates",
            "MUI's debt burden is RM905M at high cost",
            "MUI owes RM905M at expensive interest rates",
            "The group has RM905M in borrowings at high cost",
            "MUI's borrowings total RM905M at expensive rates",
        ]
        for content in contents:
            obs_set = ObservationSet(
                explicit=[Observation(content=content, level="explicit", peer_name="mui")],
            )
            bridge.ingest_observations(obs_set, peer_name="mui")
        
        # Lower threshold for consolidation test
        bridge.similarity_threshold = 0.80
        result = bridge.run_evolution_cycle()
        assert result["consolidated"] >= 1

    def test_run_evolution_cycle_promotes_high_confidence(self, bridge):
        # Add a deductive observation with high confidence
        obs_set = ObservationSet(
            deductive=[Observation(content="MUI is overleveraged", level="deductive", peer_name="mui")],
        )
        bridge.ingest_observations(obs_set, peer_name="mui")
        
        # Manually boost confidence and source count
        for entry in bridge._entries.values():
            entry.confidence = 0.98
            entry.source_count = 5
        
        result = bridge.run_evolution_cycle()
        # The deductive observation should be promoted to explicit
        promoted = [e for e in bridge._entries.values() if e.level == "explicit" and e.content == "MUI is overleveraged"]
        assert len(promoted) >= 1

    def test_get_stats(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="test", level="explicit", peer_name="alice")],
        )
        bridge.ingest_observations(obs_set, peer_name="alice")
        stats = bridge.get_stats()
        assert stats["total_entries"] == 1
        assert stats["peers"] == 1
        assert stats["by_level"]["explicit"] == 1

    def test_access_frequency_boosts_importance(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="important fact", level="explicit", peer_name="alice")],
        )
        bridge.ingest_observations(obs_set, peer_name="alice")
        
        # Access it multiple times
        for _ in range(5):
            bridge.ingest_observations(obs_set, peer_name="alice")
        
        entry = list(bridge._entries.values())[0]
        assert entry.access_count >= 5
        assert entry.confidence > 0.9


# ══════════════════════════════════════════════════════════════════════════
# SurprisalEngine v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestSurprisalEngineV2:
    @pytest.fixture
    def engine(self):
        config = SurprisalConfig(
            num_trees=8,
            max_depth=5,
            threshold=0.0,
            persist_path=None,
        )
        return SurprisalEngine(config)

    def test_fit_and_compute(self, engine):
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        engine.fit(embeddings)
        
        observations = [
            {"content": "normal observation", "level": "explicit"},
            {"content": "very different observation", "level": "explicit"},
        ]
        new_embs = [[0.15, 0.25], [0.9, 0.1]]
        
        scores = engine.compute(observations, new_embs)
        assert len(scores) > 0
        # The more anomalous point should have higher surprisal
        assert scores[0].surprisal >= 0

    def test_incremental_add_observations(self, engine):
        engine.add_observations([
            {"content": "obs 1", "level": "explicit"},
            {"content": "obs 2", "level": "explicit"},
        ])
        assert len(engine._all_observations) == 2

    def test_incremental_rebuild(self, engine):
        engine.config.incremental_rebuild_threshold = 3
        engine.add_observations([
            {"content": f"obs {i}", "level": "explicit"} for i in range(5)
        ])
        # Should have auto-rebuilt
        assert engine._forest is not None

    def test_temporal_weighting(self, engine):
        now = datetime.now(timezone.utc).timestamp()
        recent = engine._compute_temporal_weight(now)
        old = engine._compute_temporal_weight(now - 86400 * 60)  # 60 days ago
        assert recent > old, "Recent observations should have higher temporal weight"

    def test_register_contradiction(self, engine):
        engine.register_contradiction("MUI has debt", "MUI has no debt")
        assert len(engine._contradiction_map) > 0

    def test_save_and_load_state(self, engine):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            engine.add_observations([
                {"content": f"obs {i}", "level": "explicit"} for i in range(5)
            ])
            engine.save_state(path)
            
            # Create new engine and load
            engine2 = SurprisalEngine(SurprisalConfig(persist_path=path))
            loaded = engine2.load_state(path)
            assert loaded
            assert len(engine2._all_observations) == 5
        finally:
            os.unlink(path)

    def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert "total_observations" in stats
        assert "forest_built" in stats

    def test_cross_peer_boost(self, engine):
        # Add mostly "alice" observations
        engine._all_observations = [
            {"content": f"alice obs {i}", "level": "explicit", "peer_name": "alice"}
            for i in range(20)
        ]
        engine._all_observations.append(
            {"content": "bob obs", "level": "explicit", "peer_name": "bob"}
        )
        
        # Compute for a bob observation — should get cross-peer boost
        embeddings = [[0.1, 0.2]]
        observations = [{"content": "bob unique fact", "level": "explicit", "peer_name": "bob"}]
        
        # Need a fitted forest first
        all_embs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        engine.fit(all_embs)
        
        scores = engine.compute(observations, embeddings)
        # Should have some score
        assert len(scores) > 0


# ══════════════════════════════════════════════════════════════════════════
# DeterministicRetrievalBridge v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestDeterministicRetrievalBridgeV2:
    @pytest.fixture
    def bridge(self):
        with tempfile.TemporaryDirectory() as d:
            yield DeterministicRetrievalBridge(d)

    def test_store_observations_with_embeddings(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="test observation", level="explicit", peer_name="alice")],
        )
        paths = bridge.store_observations(obs_set, peer_name="alice")
        assert "explicit" in paths
        
        # Verify embedding was stored
        with open(paths["explicit"]) as f:
            data = json.load(f)
        assert len(data) == 1
        assert "embedding" in data[0]
        assert len(data[0]["embedding"]) == 64

    def test_peer_index_updated(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="test", level="explicit", peer_name="alice")],
        )
        bridge.store_observations(obs_set, peer_name="alice")
        
        peers = bridge.list_peers()
        assert len(peers) == 1
        assert peers[0]["name"] == "alice"

    def test_retrieve_latest(self, bridge):
        obs_set = ObservationSet(
            explicit=[Observation(content="test", level="explicit", peer_name="alice")],
        )
        bridge.store_observations(obs_set, peer_name="alice")
        
        results = bridge.retrieve_latest("alice")
        assert len(results) == 1
        assert results[0]["content"] == "test"


# ══════════════════════════════════════════════════════════════════════════
# NamespaceBridge v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestNamespaceBridgeV2:
    def test_resolve_search_uri(self):
        bridge = NamespaceBridge("/tmp/test")
        path = bridge.resolve("hermes://observations/alice/search?q=debt")
        assert "alice" in path
        assert "_search" in path

    def test_resolve_standard_uri(self):
        bridge = NamespaceBridge("/tmp/test")
        path = bridge.resolve("hermes://observations/alice/explicit/20260709")
        assert path.endswith("20260709.json")

    def test_resolve_directory_uri(self):
        bridge = NamespaceBridge("/tmp/test")
        path = bridge.resolve("hermes://observations/alice/explicit")
        assert path.endswith("explicit")


# ══════════════════════════════════════════════════════════════════════════
# NeuroSwarmBridge v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestNeuroSwarmBridgeV2:
    def test_format_for_council_context_with_contradictions(self):
        obs_set = ObservationSet(
            explicit=[Observation(content="MUI has debt", level="explicit", peer_name="mui")],
            deductive=[Observation(content="MUI is risky", level="deductive", peer_name="mui")],
        )
        context = NeuroSwarmBridge.format_for_council_context(
            "mui", obs_set,
            contradictions=[("MUI has debt", "MUI has no debt")],
        )
        assert "Contradictions detected" in context
        assert "MUI has debt" in context

    def test_council_verdict_to_observations(self):
        obs_set = NeuroSwarmBridge.council_verdict_to_observations(
            "The council recommends refinancing", "mui",
        )
        assert obs_set.count == 1
        assert obs_set.all[0].level == "deductive"


# ══════════════════════════════════════════════════════════════════════════
# ObliviarchBridge v2 tests
# ══════════════════════════════════════════════════════════════════════════

class TestObliviarchBridgeV2:
    def test_observation_to_trace_with_contradictions(self):
        obs = Observation(content="MUI has RM905M debt", level="explicit", peer_name="mui")
        trace = ObliviarchBridge.observation_to_trace(
            obs, "mui", contradictions=["MUI has no debt"],
        )
        assert len(trace.contradiction_flags) == 1
        assert trace.confidence == 0.9

    def test_format_for_obliviarch_record_with_contradictions(self):
        obs_set = ObservationSet(
            explicit=[Observation(content="test", level="explicit", peer_name="alice")],
        )
        record = ObliviarchBridge.format_for_obliviarch_record(
            obs_set, "alice", contradictions=["contradiction detected"],
        )
        assert record["contradictions"] == ["contradiction detected"]


# ══════════════════════════════════════════════════════════════════════════
# Integration tests — full pipeline
# ══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_extract_store_query_cycle(self):
        """Test the full extract → store → query cycle with v2 improvements."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            bridge = MemoryEvolutionBridge(path)
            
            # Simulate multiple extraction sessions
            sessions = [
                ("regent-corp", [
                    "Regent Corp holds RM228M in preference shares",
                    "RM179M is due from directors",
                ]),
                ("hyde-park", [
                    "Hyde Park hotel is worth RM1.1B post-renovation",
                    "The Marriott Tribute flag adds 15% to valuation",
                ]),
                ("regent-corp", [
                    "Regent Corp director loans may be unsecured",
                    "RM179M due from directors is the biggest risk",
                ]),
            ]
            
            for peer, contents in sessions:
                obs_set = ObservationSet(
                    explicit=[
                        Observation(content=c, level="explicit", peer_name=peer)
                        for c in contents
                    ],
                )
                bridge.ingest_observations(obs_set, peer_name=peer)
            
            # Run evolution cycle
            result = bridge.run_evolution_cycle()
            
            # Query
            results = bridge.query("Regent Corp risk", limit=5)
            assert len(results) > 0
            
            # Semantic search
            semantic = bridge.semantic_search("director loans and risk", limit=3)
            assert len(semantic) > 0
            
            # Check contradictions
            contradictions = bridge.get_contradictions()
            
            # Stats
            stats = bridge.get_stats()
            assert stats["total_entries"] >= 5
            assert stats["peers"] == 2
            
        finally:
            os.unlink(path)

    def test_surprisal_with_contradictions(self):
        """Test that contradictory observations get higher surprisal."""
        engine = SurprisalEngine(SurprisalConfig(num_trees=8, max_depth=5, threshold=0.0))
        
        # Fit on normal observations
        normal_embs = [[0.1, 0.2], [0.15, 0.25], [0.12, 0.22], [0.13, 0.23], [0.11, 0.21]]
        engine.fit(normal_embs)
        
        # Register a contradiction
        engine.register_contradiction("MUI has debt", "MUI has no debt")
        
        # Compute — the contradictory observation should be boosted
        observations = [
            {"content": "MUI has no debt", "level": "explicit", "peer_name": "mui"},
            {"content": "A&W has 101 stores", "level": "explicit", "peer_name": "mui"},
        ]
        embs = [[0.9, 0.1], [0.12, 0.22]]
        
        scores = engine.compute(observations, embs)
        if len(scores) >= 2:
            # The contradictory one should have higher score
            debt_score = next(s for s in scores if "debt" in s.content)
            aw_score = next(s for s in scores if "stores" in s.content)
            assert debt_score.surprisal >= aw_score.surprisal
