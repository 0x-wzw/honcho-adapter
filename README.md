# 🏴 Honcho Adapter

> Honcho's reasoning-first memory, extracted and bridged into your existing Hermes stack.

**No Postgres. No server. No AGPL. Just the core reasoning engine.**

This repo extracts the three capabilities that make Honcho unique — the **deriver** (LLM-based observation extraction), the **dialectic** (agentic query answering), and the **surprisal engine** (novelty detection) — and bridges them into your existing memory stack: memory-evolution, OBLIVIARCH, deterministic-retrieval, recursive-retrieval, hermes-namespace, and NEUROSWARM.

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │           honcho-adapter              │
                         │  MIT · stdlib-first · no server      │
                         └─────────────────────────────────────┘

    ┌───────────────────┐  ┌───────────────┐  ┌───────────────────┐
    │    Deriver         │  │  Dialectic    │  │   Surprisal       │
    │  LLM observation   │  │  Agentic QA   │  │  Novelty detection│
    │  extraction        │  │  with tools    │  │  via RP forest    │
    └────────┬──────────┘  └──────┬────────┘  └────────┬──────────┘
             │                    │                     │
             ▼                    ▼                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                     Bridges Layer                           │
    │  Converts honcho-adapter types ↔ each repo's native format │
    └────────────────────────────────────────────────────────────┘
             │                    │                     │
    ┌────────┴────────┐  ┌───────┴────────┐  ┌────────┴────────┐
    │ memory-evolution │  │   OBLIVIARCH   │  │ deterministic-  │
    │ • Store obs      │  │ • Compress obs │  │   retrieval     │
    │ • Decay/rewrite  │  │ • Schema/arch  │  │ • Path-based    │
    │ • Relationship   │  │ • 500x         │  │ • Reproducible  │
    └─────────────────┘  └────────────────┘  └─────────────────┘
             │                    │                     │
    ┌────────┴────────┐  ┌───────┴────────┐  ┌────────┴────────┐
    │ recursive-       │  │   hermes-      │  │   NEUROSWARM    │
    │ retrieval        │  │   namespace    │  │ • GBrain (WHAT) │
    │ • L0→L1→L2 merge │  │ • URI resolve  │  │ • NecroSwarm    │
    │ • Overlay/combine│  │ • obs:// obs   │  │   (HOW)         │
    └─────────────────┘  └────────────────┘  └─────────────────┘
```

---

## Why This Exists

Your existing stack is already more sophisticated than Honcho in every dimension **except one**: LLM-based content understanding.

| What Your Stack Does | What Honcho Adds |
|---|---|
| Store raw text (JSON) | Extract structured **conclusions** from conversations |
| Retrieve by path (deterministic) | Answer "what does the system know about X?" in natural language |
| Merge context layers (recursive) | Model what **one agent knows about another** |
| Compress patterns (OBLIVIARCH) | Find **contradictory** observations across sessions |
| Decay and rewrite (memory-evolution) | Surface **novel/interesting** patterns (surprisal) |
| Deliberate via council (NEUROSWARM) | Feed observations into the 10-D council as context |

**honcho-adapter fills the single gap:** reading conversations and extracting structured conclusions using an LLM, then feeding those conclusions into your existing stack.

---

## Quick Start

### Install

```bash
pip install honcho-adapter

# Or with your preferred LLM provider:
pip install "honcho-adapter[openai]"    # OpenAI / compatible
pip install "honcho-adapter[anthropic]" # Anthropic Claude
pip install "honcho-adapter[gemini]"    # Google Gemini
```

### 1. Extract Observations from a Conversation

```python
import asyncio
from honcho_adapter import create_deriver

async def main():
    deriver = create_deriver(
        provider="openai",
        model="gpt-4o-mini",
    )

    # Extract observations from a conversation
    obs = await deriver.extract(
        peer_name="alice",
        messages=[
            {"peer": "alice", "content": "I just turned 25!", "timestamp": "2026-07-09"},
            {"peer": "bob", "content": "Happy birthday! Any plans?", "timestamp": "2026-07-09"},
            {"peer": "alice", "content": "I'm going to NYC for a week. Taking my dog.", "timestamp": "2026-07-09"},
        ],
    )

    print(f"Extracted {obs.count} observations")
    for o in obs.explicit:
        print(f"  📌 {o.content}")
    for o in obs.deductive:
        print(f"  🧠 {o.content}")

asyncio.run(main())
```

### 2. Store in memory-evolution

```python
from honcho_adapter.bridges import MemoryEvolutionBridge

bridge = MemoryEvolutionBridge("memory_store.json")
bridge.ingest_observations(obs, peer_name="alice", session_name="chat-1")

# Query later
results = bridge.query("alice birthday")
for r in results:
    print(f"  [{r.level}] {r.content}")
```

### 3. Feed into OBLIVIARCH

```python
from honcho_adapter.bridges import ObliviarchBridge

trace = ObliviarchBridge.observation_to_trace(obs.all[0], peer_name="alice")
# Now feed trace into OBLIVIARCH's record() API:
#   obliviarch.record(**trace.__dict__)
```

### 4. Query with the Dialectic Agent

```python
from honcho_adapter import create_dialectic

agent = create_dialectic(
    observer="system",
    observed="alice",
    provider="openai",
    model="gpt-4o-mini",
)

# Register memory tools
agent.add_tool(
    "search_memory",
    lambda query: str(bridge.query(query)),
    "Search observations about the peer",
)

answer = await agent.answer("What does Alice like to do in her free time?")
print(answer)
```

### 5. Find Surprising Observations

```python
from honcho_adapter import SurprisalEngine
import numpy as np

# Build embeddings from existing observations
existing_embs = [np.random.randn(384).tolist() for _ in range(100)]
engine = SurprisalEngine()
engine.fit(existing_embs)

# Find surprising new observations
new_obs = [{"content": "alice adopted a pet crocodile", "level": "explicit"}]
new_embs = [np.random.randn(384).tolist()]
scores = engine.compute(new_obs, new_embs)
for s in scores:
    print(f"  Surprisal: {s.surprisal:.3f} — {s.content}")
```

---

## Integration Guides

### With memory-evolution

The `MemoryEvolutionBridge` stores observations in memory-evolution's JSON format, then runs decay and relationship inference. Observations with `level="explicit"` get high importance; `level="deductive"` observations get relationship links to their source observations.

```python
from honcho_adapter.bridges import MemoryEvolutionBridge

# All-in-one
bridge = MemoryEvolutionBridge()
bridge.ingest_observations(obs, peer_name="alice")
bridge.run_evolution_cycle()  # decay + relationship inference
```

### With OBLIVIARCH

The `ObliviarchBridge` converts observations to episodic traces. High-frequency observation patterns get promoted to schemas (~20x compression), then archetypes (~500x compression).

```python
from honcho_adapter.bridges import ObliviarchBridge

trace_data = ObliviarchBridge.format_for_obliviarch_record(obs, peer_name="alice")
# obliviarch.record(**trace_data)
```

### With deterministic-retrieval

The `DeterministicRetrievalBridge` writes observations to a structured directory tree:

```
~/.hermes/workspace/observations/
├── alice/
│   ├── explicit/
│   │   ├── 20260709.json
│   │   └── 20260710.json
│   └── deductive/
│       └── 20260709.json
├── bob/
│   └── explicit/
│       └── 20260709.json
```

These files are accessible via `hermes-retrieve --path observations/alice/explicit/20260709.json`.

### With recursive-retrieval

The `RecursiveRetrievalBridge` converts observations to an agent-layer (L1) context dict:

```python
from honcho_adapter.bridges import RecursiveRetrievalBridge

layer = RecursiveRetrievalBridge.observations_to_agent_layer("alice", obs.explicit, obs.deductive)
# context_tree.add_layer("agent", layer)
```

### With hermes-namespace

The `NamespaceBridge` registers `hermes://observations/...` URIs:

```
hermes://observations/alice               → ~/.hermes/workspace/observations/alice
hermes://observations/alice/explicit      → ~/.hermes/workspace/observations/alice/explicit
hermes://observations/alice/explicit/20260709  → ~/.../observations/alice/explicit/20260709.json
```

### With NEUROSWARM

The `NeuroSwarmBridge` wraps the dialectic agent as a NEUROSWARM tool and formats observations for the 10-D council:

```python
from honcho_adapter.bridges import NeuroSwarmBridge

# Wrap as a tool
tool = NeuroSwarmBridge.create_dialectic_tool(agent)
# Register in NEUROSWARM's tool registry

# Format for council deliberation
context = NeuroSwarmBridge.format_for_council_context("alice", obs)
# Feed into council: council.deliberate(task, context=context)
```

---

## Comparison: Upstream Honcho vs honcho-adapter

| Dimension | Upstream Honcho (AGPL) | honcho-adapter (MIT) |
|-----------|----------------------|---------------------|
| **License** | AGPL-3.0 (strong copyleft) | MIT (no restrictions) |
| **Server** | FastAPI + Postgres + Redis | None — pure Python library |
| **Dependencies** | 30+ (FastAPI, SQLAlchemy, pgvector, etc.) | Minimal (numpy, scikit-learn) |
| **LLM Providers** | OpenAI, Anthropic, Gemini | Same (optional install) |
| **Deriver** | Reads from Postgres, writes to Postgres | Reads from Python dicts, writes to any store |
| **Dialectic** | Postgres-backed queries, server endpoints | Tool-based, plugs into any memory backend |
| **Surprisal** | Postgres + sklearn trees | numpy + sklearn (same math, no DB) |
| **Dreamer** | Full dream cycle with specialists | Not included — OBLIVIARCH replaces this |
| **Peer Model** | Native `(observer, observed)` collections | Bridges map to your existing peer model |
| **Telemetry** | Prometheus, Sentry, Langfuse, CloudEvents | Not included — use your existing monitoring |
| **Auth** | JWT, scoped API keys | Not included — use your existing auth |
| **Offline** | No (requires server) | Yes (stdlib-only) |
| **Deployment** | Docker / Fly.io | `pip install` |

---

## v2.0 — Memory That Performs Better Over Time

v2.0 adds a suite of improvements that make the memory block smarter as it accumulates observations:

### Deriver v2
- **Confidence scoring** — each observation gets a 0.0-1.0 confidence score
- **Contradiction detection** — detects contradictions within a single extraction pass
- **Entity extraction** — extracts named entities (people, orgs, places, metrics)
- **Sentiment analysis** — per-observation sentiment (positive, negative, neutral)
- **Source tracking** — maps observations back to source message IDs
- **Deduplication** — removes exact duplicates within a single pass

### Dialectic v2
- **Structured JSON tool protocol** — not fragile `[TOOL:...]` text markers
- **Response caching** — same query → cached answer (TTL + observation-count invalidation)
- **Source citations** — answers include which observations support them
- **Confidence scoring** — per-answer confidence (0.0-1.0)
- **Answer history** — tracks past answers for follow-up context

### Surprisal Engine v2
- **Real embedding-based features** — deterministic n-gram hashing (not proxy features)
- **Incremental fitting** — add observations without rebuilding from scratch
- **Temporal weighting** — recent observations weighted higher for novelty
- **Cross-peer analysis** — finds surprising patterns across different peers
- **Contradiction-aware scoring** — contradictory observations get boosted
- **Persistence** — save/load forest state to disk

### MemoryEvolutionBridge v2
- **Embedding-based semantic search** — not just text match
- **Dedup via semantic similarity** — within same level, configurable threshold
- **Consolidation** — merges similar observations into higher-level insights
- **Access-frequency-boosted decay** — frequently accessed memories decay slower
- **Contradiction detection** — flags contradictory observations across sessions
- **Temporal weighting** — recent observations score higher
- **Confidence promotion** — high-confidence deductive observations promoted to explicit

### DeterministicRetrievalBridge v2
- **Embeddings stored alongside observations** — for semantic retrieval
- **Peer index** — fast listing of peers with observation counts
- **Latest retrieval** — get most recent observations for a peer

### NamespaceBridge v2
- **Search URI support** — `hermes://observations/{peer}/search?q=...`

### NeuroSwarmBridge v2
- **Contradiction context** — includes contradiction warnings in council context
- **Confidence indicators** — high/medium tags per observation

### ObliviarchBridge v2
- **Richer trace metadata** — confidence, contradiction flags, source count

---

## Repo Structure

```
honcho-adapter/
├── LICENSE                     # MIT
├── pyproject.toml              # Package config
├── README.md                   # ← You are here
├── honcho_adapter/
│   ├── __init__.py             # Package root
│   ├── llm.py                  # LLM provider abstraction (OpenAI, Anthropic, Gemini)
│   ├── deriver.py              # LLM-based observation extraction from conversations
│   ├── dialectic.py            # Agentic query answering with tool-based context gathering
│   ├── surprisal.py            # Novelty detection via RP forest (geometric surprisal)
│   └── bridges.py              # → memory-evolution, OBLIVIARCH, deterministic-retrieval,
│                                #   recursive-retrieval, hermes-namespace, NEUROSWARM
├── tests/
│   ├── test_deriver.py
│   ├── test_dialectic.py
│   ├── test_surprisal.py
│   └── test_bridges.py
└── examples/
    ├── quickstart.py
    └── full_pipeline.py
```

---

## Upstream

This repo is adapted from [plastic-labs/honcho](https://github.com/plastic-labs/honcho) (v3.0.11) — the deriver prompt, dialectic system prompt, and surprisal tree algorithm are derived from Honcho's source under AGPL-3.0. The adapter reimplements these as a standalone MIT library by stripping all Postgres, server, and telemetry dependencies.

The forked Honcho repo lives at [0x-wzw/honcho](https://github.com/0x-wzw/honcho) with the adapter integration branch.

---

## License

MIT — Free to use, modify, and distribute. No copyleft restrictions.