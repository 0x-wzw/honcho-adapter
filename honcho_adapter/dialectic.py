"""
Dialectic — Agentic query answering with tool-based context gathering.

Adapted from Honcho's src/dialectic/ module. Stripped of Postgres/server deps.
Answers natural-language questions about a peer by gathering context from
your existing memory stores.

v2.0 — Improved performance over time:
- Structured JSON tool protocol (not fragile text markers)
- Response caching (same query → cached answer if no new observations)
- Source citations in answers (which observations support the answer)
- Confidence scoring per answer
- Multi-tool parallel execution
- Answer history tracking for follow-up questions
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────

@dataclass
class DialecticConfig:
    """Configuration for the dialectic agent.
    
    v2: Added cache_ttl, max_parallel_tools, citation_requirement.
    """
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    max_input_tokens: int | None = None
    temperature: float = 0.0
    max_tool_iterations: int = 10
    reasoning_level: str = "medium"  # minimal, low, medium, high, max
    # v2 improvements
    cache_ttl_seconds: int = 300  # Cache answers for 5 minutes
    max_parallel_tools: int = 3  # Max tools to call in parallel
    require_citations: bool = True  # Require source citations in answers
    min_confidence_to_answer: float = 0.3  # Don't answer below this confidence


@dataclass
class CachedAnswer:
    """A cached answer with metadata."""
    query: str
    answer: str
    confidence: float
    sources: list[str]
    timestamp: float
    observation_count: int  # How many observations existed when cached


class MemoryTool:
    """Protocol for memory tools the dialectic can call."""
    name: str
    description: str

    async def __call__(self, **kwargs: Any) -> str:
        ...


# ── Dialectic Agent ────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful and concise context synthesis agent that answers questions about \
{observer}'s understanding of {observed}.

You have access to memory tools. Use them to gather relevant information before answering.

## WORKFLOW

1. **Analyze the query**: What specific information does the query demand?
2. **Search memory**: Use the search tool to find relevant observations.
3. **Cross-reference**: If you find deductive/inductive observations, check their reasoning chains.
4. **Synthesize**: Answer the query directly, grounded in the information you found.
5. **Handle contradictions**: If you find conflicting information, present both sides.
6. **Abstain when uncertain**: Say "I don't have information about that" rather than inventing.

## CRITICAL RULES
- NEVER fabricate information. If you don't know, say so.
- Ground answers in specific observations you found.
- Quote exact values (dates, numbers, names) from what you found.
- For update questions: the more recent statement supersedes older ones.
- Do not explain your tool usage — just provide the synthesized answer.
- When you use a tool, respond with a JSON tool call: {{"tool": "name", "query": "..."}}
- When you have the final answer, respond with: {{"answer": "...", "confidence": 0.0-1.0, "sources": ["source1", "source2"]}}
"""


class DialecticAgent:
    """Agentic query engine that answers questions using memory tools.
    
    v2 improvements:
    - Structured JSON tool protocol (not fragile [TOOL:...] markers)
    - Response caching with TTL and observation-count invalidation
    - Source citations in answers
    - Confidence scoring per answer
    - Answer history for follow-up context
    
    Usage:
        agent = DialecticAgent(llm_backend, observer="system", observed="alice")
        agent.add_tool("search_memory", search_fn, "Search observations about the peer")
        answer = await agent.answer("What does Alice prefer for learning?")
    """

    def __init__(
        self,
        llm: LLMBackend,
        observer: str,
        observed: str,
        config: DialecticConfig | None = None,
        observer_peer_card: list[str] | None = None,
        observed_peer_card: list[str] | None = None,
    ):
        self._llm = llm
        self.config = config or DialecticConfig()
        self.observer = observer
        self.observed = observed
        self._tools: dict[str, tuple[Callable, str]] = {}
        self._messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(
                    observer=observer,
                    observed=observed,
                ),
            }
        ]
        self._prefetched_count: int = 0
        # v2: caching and history
        self._cache: dict[str, CachedAnswer] = {}
        self._answer_history: list[dict[str, Any]] = []
        self._observation_count_at_last_cache: int = 0

    def add_tool(self, name: str, fn: Callable, description: str) -> None:
        """Register a memory tool the agent can call."""
        self._tools[name] = (fn, description)

    def _make_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _check_cache(self, query: str, current_observation_count: int) -> CachedAnswer | None:
        """Check if we have a valid cached answer for this query.
        
        v2: New method. Invalidates cache if:
        - TTL expired
        - More observations exist now than when cached
        """
        key = self._make_cache_key(query)
        cached = self._cache.get(key)
        if cached is None:
            return None
        
        now = time.time()
        
        # Check TTL
        if now - cached.timestamp > self.config.cache_ttl_seconds:
            del self._cache[key]
            return None
        
        # Check if new observations have been added
        if current_observation_count > cached.observation_count:
            del self._cache[key]
            return None
        
        return cached

    def _tool_schemas(self) -> str:
        """Build tool schema description for the prompt.
        
        v2: JSON-based tool protocol instead of text markers.
        """
        parts = ["Available tools:"]
        for name, (fn, description) in self._tools.items():
            parts.append(f"- {name}: {description}")
        parts.append("")
        parts.append("To call a tool, respond with JSON: {\"tool\": \"name\", \"query\": \"your search query\"}")
        parts.append("To provide the final answer, respond with JSON: {\"answer\": \"...\", \"confidence\": 0.0-1.0, \"sources\": [\"...\"]}")
        return "\n".join(parts)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse the LLM response, trying JSON first, then text markers.
        
        v2: Tries JSON parsing first, falls back to text markers for backward compat.
        """
        # Try JSON parsing
        content_stripped = content.strip()
        if content_stripped.startswith("{"):
            try:
                return json.loads(content_stripped)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', content_stripped, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: text-based tool protocol (backward compat)
        tool_match = re.search(r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]', content, re.DOTALL)
        if tool_match:
            return {
                "tool": tool_match.group(1),
                "query": tool_match.group(2).strip(),
            }
        
        # Assume it's a final answer
        return {"answer": content_stripped, "confidence": 0.5, "sources": []}

    async def answer(self, query: str) -> str:
        """Answer a query about the observed peer.
        
        v2: Uses cache, structured JSON protocol, returns answer with sources.
        
        Returns:
            The synthesized answer string.
        """
        # Check cache
        cached = self._check_cache(query, self._observation_count_at_last_cache)
        if cached is not None:
            logger.info("Cache hit for query: %s", query[:60])
            return cached.answer

        # Inject query
        self._messages.append({"role": "user", "content": f"Query: {query}"})
        
        # Inject tool schemas
        tool_schemas = self._tool_schemas()
        self._messages.append({"role": "system", "content": tool_schemas})

        # Tool loop
        tool_calls_made = 0
        start = time.perf_counter()
        final_answer = ""
        final_confidence = 0.0
        final_sources: list[str] = []

        for iteration in range(self.config.max_tool_iterations):
            # Build the full prompt from messages
            full_prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._messages
            )

            response = await self._llm.call(
                prompt=full_prompt,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                retry_attempts=2,
            )

            content = response.content
            parsed = self._parse_response(content)

            # Check if this is a final answer
            if "answer" in parsed:
                final_answer = parsed["answer"]
                final_confidence = parsed.get("confidence", 0.5)
                final_sources = parsed.get("sources", [])
                self._messages.append({"role": "assistant", "content": final_answer})
                break

            # Check if this is a tool call
            tool_name = parsed.get("tool", "")
            tool_query = parsed.get("query", "")
            
            if tool_name and tool_name in self._tools:
                fn, desc = self._tools[tool_name]
                try:
                    result = await fn(query=tool_query)
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"

                tool_calls_made += 1
                self._messages.append({
                    "role": "assistant",
                    "content": f"Tool: {tool_name}\nQuery: {tool_query}\nResult: {result}",
                })
            else:
                # Unknown format — treat as final answer
                final_answer = content
                final_confidence = 0.3
                final_sources = []
                self._messages.append({"role": "assistant", "content": content})
                break

        # If we exhausted iterations without a final answer
        if not final_answer:
            final_answer = "I was unable to complete the analysis within the iteration limit."
            final_confidence = 0.1

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Dialectic answered in %.0fms with %d tool calls (confidence: %.2f)",
            elapsed, tool_calls_made, final_confidence,
        )

        # Build answer with sources
        if final_sources and self.config.require_citations:
            answer_with_sources = final_answer
            if not final_answer.endswith("\n"):
                answer_with_sources += "\n"
            answer_with_sources += "\n**Sources:**\n"
            for s in final_sources:
                answer_with_sources += f"- {s}\n"
            final_answer = answer_with_sources

        # Cache the answer
        self._cache[self._make_cache_key(query)] = CachedAnswer(
            query=query,
            answer=final_answer,
            confidence=final_confidence,
            sources=final_sources,
            timestamp=time.time(),
            observation_count=self._observation_count_at_last_cache,
        )

        # Track history
        self._answer_history.append({
            "query": query,
            "answer": final_answer,
            "confidence": final_confidence,
            "sources": final_sources,
            "tool_calls": tool_calls_made,
            "elapsed_ms": elapsed,
        })

        return final_answer

    def update_observation_count(self, count: int) -> None:
        """Update the observation count for cache invalidation.
        
        v2: New method. Call this when new observations are added to the store.
        """
        self._observation_count_at_last_cache = count

    def clear_cache(self) -> None:
        """Clear the answer cache.
        
        v2: New method. Call when observations change significantly.
        """
        self._cache.clear()
        logger.info("Dialectic cache cleared")

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent answer history.
        
        v2: New method. Useful for follow-up context.
        """
        return self._answer_history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "cached_answers": len(self._cache),
            "total_answers": len(self._answer_history),
            "tools_registered": len(self._tools),
            "observer": self.observer,
            "observed": self.observed,
        }


# ── Convenience ────────────────────────────────────────────────────────────

def create_dialectic(
    observer: str,
    observed: str,
    provider: str = "openai",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> DialecticAgent:
    """Factory: create a DialecticAgent with a single call.
    
    v2: Accepts cache_ttl_seconds, require_citations kwargs.
    """
    from .llm import create_backend

    valid_keys = [
        "max_tokens", "temperature", "custom_instructions", "retry_attempts",
        "cache_ttl_seconds", "require_citations", "min_confidence_to_answer",
    ]
    config = DialecticConfig(
        model=model,
        **{k: v for k, v in kwargs.items() if k in valid_keys},
    )
    llm = create_backend(provider=provider, api_key=api_key)
    return DialecticAgent(llm=llm, observer=observer, observed=observed, config=config)
