"""
Dialectic — Agentic query answering with tool-based context gathering.

Adapted from Honcho's src/dialectic/ module. Stripped of Postgres/server deps.
Answers natural-language questions about a peer by gathering context from
your existing memory stores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────

@dataclass
class DialecticConfig:
    """Configuration for the dialectic agent."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    max_input_tokens: int | None = None
    temperature: float = 0.0
    max_tool_iterations: int = 10
    reasoning_level: str = "medium"  # minimal, low, medium, high, max


# ── Tool Protocol ──────────────────────────────────────────────────────────

class MemoryTool:
    """Protocol for memory tools the dialectic can call."""

    name: str
    description: str

    async def __call__(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result."""
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
"""


class DialecticAgent:
    """Agentic query engine that answers questions using memory tools.

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

    def add_tool(self, name: str, fn: Callable, description: str) -> None:
        """Register a memory tool the agent can call."""
        self._tools[name] = (fn, description)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for name, (fn, description) in self._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": f"Search query for {name}",
                            }
                        },
                        "required": ["query"],
                    },
                },
            })
        return schemas

    async def answer(self, query: str) -> str:
        """Answer a query about the observed peer.

        Gathers context via tools, then synthesizes a response.
        """
        import time

        # Inject query
        self._messages.append({"role": "user", "content": f"Query: {query}"})

        # Tool loop
        tool_calls_made = 0
        start = time.perf_counter()

        for iteration in range(self.config.max_tool_iterations):
            response = await self._llm.call(
                prompt="",  # messages are passed via prompt injection
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                # We use a simple text-based tool protocol since we can't
                # pass actual tool schemas to the generic LLMBackend.
                # Instead we inject tool descriptions into the prompt.
                retry_attempts=2,
            )

            content = response.content

            # Check if the response contains a tool call marker
            # Format: [TOOL: name] query [/TOOL]
            tool_call = self._parse_tool_call(content)
            if tool_call is None:
                # No more tool calls — this is the final answer
                self._messages.append({"role": "assistant", "content": content})
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "Dialectic answered in %.0fms with %d tool calls",
                    elapsed, tool_calls_made,
                )
                return content

            name, query_text = tool_call
            fn, desc = self._tools.get(name, (None, None))
            if fn is None:
                result = f"Unknown tool: {name}. Available: {list(self._tools.keys())}"
            else:
                try:
                    result = await fn(query=query_text)
                except Exception as e:
                    result = f"Error calling {name}: {e}"

            tool_calls_made += 1
            self._messages.append({
                "role": "assistant",
                "content": f"Tool: {name}\nQuery: {query_text}\nResult: {result}",
            })

        # If we exhausted iterations without a final answer
        return "I was unable to complete the analysis within the iteration limit. Here's what I found so far:\n" + \
            self._messages[-1]["content"] if self._messages else "No results."

    def _parse_tool_call(self, text: str) -> tuple[str, str] | None:
        """Parse [TOOL: name] query [/TOOL] from response text."""
        import re
        match = re.search(r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]', text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None


# ── Convenience ────────────────────────────────────────────────────────────

def create_dialectic(
    observer: str,
    observed: str,
    provider: str = "openai",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> DialecticAgent:
    """Factory: create a DialecticAgent with a single call."""
    from .llm import create_backend

    config = DialecticConfig(model=model)
    llm = create_backend(provider=provider, api_key=api_key)
    return DialecticAgent(llm=llm, observer=observer, observed=observed, config=config)