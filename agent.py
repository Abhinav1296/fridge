"""Tool-using "Chef" agent for SmartFridge Vision (Phase 1).

This is the conversational brain of the app. A user can ask, in plain language, "what
can I cook tonight without shopping?" or "plan three days of dinners and use up the
paneer first", and the agent answers by **calling the app's own functions** — the live
inventory, the recommender, and the zero-waste meal-plan optimizer — rather than
hallucinating. It is a small, honest ReAct-style loop: the model decides which tool to
call, we run the real Python function, feed the result back, and repeat until the model
has enough to answer.

Two execution modes, chosen automatically:

* **Native tool-calling** (``mode="tools"``) — the model returns OpenAI-style
  ``tool_calls`` and we dispatch them. This is the primary path.
* **JSON-action fallback** (``mode="json"``) — for the many *free* chat models that don't
  support function-calling, the system prompt asks the model to reply with a single JSON
  object (``{"action": ..., "arguments": {...}}`` or ``{"action": "final", ...}``) which
  we parse and dispatch identically. ``mode="auto"`` (the default) tries native tools and
  transparently drops to JSON if the provider rejects the ``tools`` parameter.

Configuration mirrors :mod:`vision_service` but reads ``LLM_*`` first, falling back to the
``VISION_*`` settings so a single key works out of the box (read from the environment at
call time; never hardcoded):

* ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` (each falls back to its ``VISION_*``
  equivalent), plus optional ``LLM_MODEL_2`` / ``LLM_MODEL_3`` model fallbacks.

Like the rest of the core, this module is **framework-agnostic** (no Flask, no HTTP-server
coupling). The concrete tools are wired to :mod:`storage`, :mod:`recommender`, and
:mod:`optimizer` in :func:`build_default_tools`, but :func:`run_agent` accepts any tool
mapping, so it is fully unit-testable offline with stub tools and a stubbed transport.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

import optimizer
import recommender

logger = logging.getLogger(__name__)

# Defaults for the agent's chat model. Overridable via LLM_* / VISION_* (see module doc).
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

_MAX_MODELS = 3            # primary + 2 model fallbacks
_MAX_STEPS = 5            # tool-call rounds before we force a final answer
_REQUEST_TIMEOUT: tuple[int, int] = (10, 90)


# --- Exceptions -------------------------------------------------------------


class AgentError(Exception):
    """Base class for agent failures."""


class AgentConfigError(AgentError):
    """No API key is configured for the agent (nor a VISION_* fallback)."""


class AgentAPIError(AgentError):
    """The chat provider was unreachable or returned an error."""


class _ToolsUnsupported(Exception):
    """Internal signal: the provider rejected native tool-calling — retry in JSON mode."""


# --- A tool: schema + implementation ----------------------------------------


@dataclass(frozen=True)
class Tool:
    """One callable tool exposed to the model.

    Attributes:
        name: The function name the model calls.
        description: What the tool does (shown to the model).
        parameters: JSON Schema for the arguments (OpenAI ``function.parameters`` shape).
        run: The implementation — ``run(arguments: dict) -> JSON-serializable result``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[Mapping[str, Any]], Any]

    def openai_schema(self) -> dict[str, Any]:
        """Return this tool in OpenAI ``tools`` array format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --- Default tool set (wired to the real app modules) -----------------------


def build_default_tools(
    *,
    get_latest_scan: Callable[[], Mapping[str, Any] | None],
    list_perishables: Callable[[], Sequence[Mapping[str, Any]]],
    list_nutrition: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Tool]:
    """Build the standard Chef tools, injecting the data-access callables.

    The web layer passes :func:`storage.get_latest_scan`, :func:`storage.list_perishables`,
    and (optionally) :func:`storage.list_nutrition`. Injecting them keeps this module free
    of a hard storage dependency and makes the agent trivially testable with fakes.
    """

    def _inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scan = get_latest_scan() or {}
        items = list(scan.get("items") or [])
        perishables = list(list_perishables() or [])
        return items, perishables

    def _nutrition_map() -> dict[str, dict[str, Any]]:
        if not list_nutrition:
            return {}
        return {row["item"]: dict(row) for row in (list_nutrition() or []) if row.get("item")}

    def get_inventory(_args: Mapping[str, Any]) -> dict[str, Any]:
        items, perishables = _inventory()
        return {
            "items": [
                {
                    "name": it.get("name"),
                    "count": it.get("count"),
                    "freshness": it.get("freshness"),
                }
                for it in items
            ],
            "tracked_use_by": [
                {"name": p.get("name"), "use_by": p.get("use_by")} for p in perishables
            ],
            "item_count": len(items),
        }

    def get_recommendations(_args: Mapping[str, Any]) -> dict[str, Any]:
        items, perishables = _inventory()
        rec = recommender.recommend(items, perishables)
        # Trim to what the model needs to reason and answer (keeps the context small).
        return {
            "use_soon": rec["use_soon"],
            "recipes": [
                {"title": r["title"], "missing": r["missing"], "can_make": r["can_make"]}
                for r in rec["recipes"]
            ],
            "shopping": rec["shopping"],
        }

    def plan_meals(args: Mapping[str, Any]) -> dict[str, Any]:
        items, perishables = _inventory()
        days = _as_int(args.get("days"), 3, lo=1, hi=7)
        meals = _as_int(args.get("meals_per_day"), 2, lo=1, hi=4)
        plan = optimizer.plan_meals(
            items,
            perishables,
            days=days,
            meals_per_day=meals,
            nutrition=_nutrition_map() or None,
        )
        # Return the decision-relevant fields (drop internal scoring detail).
        return {
            "solver": plan["solver"],
            "horizon": plan["horizon"],
            "plan": [
                {
                    "title": row["title"],
                    "uses": row["uses"],
                    "uses_expiring": row["uses_expiring"],
                    "to_buy": row["to_buy"],
                }
                for row in plan["plan"]
            ],
            "shopping_list": plan["shopping_list"],
            "metrics": plan["metrics"],
        }

    tools = [
        Tool(
            name="get_inventory",
            description="Get the current fridge inventory (from the most recent scan) plus "
            "any user-tracked 'use by' dates. Call this first to see what's on hand.",
            parameters={"type": "object", "properties": {}},
            run=get_inventory,
        ),
        Tool(
            name="get_recommendations",
            description="Get recipe suggestions, use-soon items, and shopping ideas from the "
            "self-trained recommender, based on the current fridge.",
            parameters={"type": "object", "properties": {}},
            run=get_recommendations,
        ),
        Tool(
            name="plan_meals",
            description="Run the zero-waste meal-plan optimizer. Returns a concrete schedule "
            "of dishes that uses up expiring food first and minimizes what must be bought, "
            "plus a combined shopping list and metrics (waste_avoided_pct, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Plan horizon in days (1-7).",
                        "minimum": 1,
                        "maximum": 7,
                    },
                    "meals_per_day": {
                        "type": "integer",
                        "description": "Meals to plan per day (1-4).",
                        "minimum": 1,
                        "maximum": 4,
                    },
                },
            },
            run=plan_meals,
        ),
    ]
    return {tool.name: tool for tool in tools}


# --- System prompts ---------------------------------------------------------


_SYSTEM_BASE = (
    "You are Chef, the assistant inside a SmartFridge app. You help the user cook what they "
    "already have and waste less food. You are grounded and honest: base every claim on the "
    "tool results, never invent inventory or recipes, and if a tool shows the fridge is "
    "empty, say so. Prefer plans that use up expiring items and need little or no shopping. "
    "Keep answers short, concrete, and friendly; use the user's own ingredient names."
)

_JSON_PROTOCOL = (
    "\n\nYou do NOT have native function-calling. Instead, to use a tool, reply with ONLY a "
    "single JSON object (no prose, no code fences) of the form:\n"
    '{"action": "<tool_name>", "arguments": { ... }}\n'
    "After you receive the tool result (as a user message), you may call another tool or "
    "give your final answer. To give the final answer, reply with ONLY:\n"
    '{"action": "final", "answer": "<your natural-language answer>"}\n'
    "Available tools:\n"
)


def _json_tools_doc(tools: Mapping[str, Tool]) -> str:
    lines = []
    for tool in tools.values():
        props = tool.parameters.get("properties", {})
        arg_names = ", ".join(props) if props else "(none)"
        lines.append(f"- {tool.name}({arg_names}): {tool.description}")
    return "\n".join(lines)


# --- Public API -------------------------------------------------------------


def configured() -> bool:
    """Return True if the agent has a usable API key (LLM_* or VISION_* fallback)."""
    return bool(_load_models())


def run_agent(
    message: str,
    *,
    tools: Mapping[str, Tool],
    history: Sequence[Mapping[str, str]] | None = None,
    mode: str = "auto",
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Answer ``message`` by letting the model call ``tools`` until it can respond.

    Args:
        message: The user's question.
        tools: Mapping of tool name -> :class:`Tool` (see :func:`build_default_tools`).
        history: Optional prior turns as ``[{"role": "user"|"assistant", "content": str}]``.
        mode: ``"auto"`` (native tools, fall back to JSON), ``"tools"``, or ``"json"``.
        max_steps: Maximum tool-call rounds before a final answer is forced.

    Returns:
        ``{"answer": str, "steps": [{"tool", "arguments", "result"}...], "mode": str,
        "model": str}``.

    Raises:
        AgentConfigError: No API key configured.
        AgentAPIError: The provider failed on every configured model.
    """
    models = _load_models()
    if not models:
        raise AgentConfigError(
            "No agent API key configured. Set LLM_API_KEY (or VISION_API_KEY) in your .env."
        )

    if mode == "json":
        return _converse(message, tools, history, models, use_tools=False, max_steps=max_steps)
    if mode == "tools":
        return _converse(message, tools, history, models, use_tools=True, max_steps=max_steps)

    # auto: try native tool-calling; fall back to JSON if the provider rejects it.
    try:
        return _converse(message, tools, history, models, use_tools=True, max_steps=max_steps)
    except _ToolsUnsupported:
        logger.info("Provider rejected native tool-calling; retrying in JSON-action mode.")
        return _converse(message, tools, history, models, use_tools=False, max_steps=max_steps)


# --- Controller loop --------------------------------------------------------


def _converse(
    message: str,
    tools: Mapping[str, Tool],
    history: Sequence[Mapping[str, str]] | None,
    models: Sequence[_Model],
    *,
    use_tools: bool,
    max_steps: int,
) -> dict[str, Any]:
    """Shared ReAct loop for both native-tools and JSON-action modes."""
    system = _SYSTEM_BASE
    if not use_tools:
        system = _SYSTEM_BASE + _JSON_PROTOCOL + _json_tools_doc(tools)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in history or []:
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            messages.append({"role": role, "content": str(turn["content"])})
    messages.append({"role": "user", "content": message})

    tool_schemas = [t.openai_schema() for t in tools.values()] if use_tools else None
    steps: list[dict[str, Any]] = []

    for _ in range(max(1, max_steps)):
        reply = _chat(models, messages, tool_schemas)

        if use_tools:
            calls = reply.get("tool_calls") or []
            if not calls:
                content = (reply.get("content") or "").strip()
                if content:
                    return _done(content, steps, "tools", reply["_model"])
                break  # model stalled with an empty, tool-less reply — force a final answer
            # Record the assistant's tool-call turn, then run each call.
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call.get("function", {}).get("name", "")
                raw_args = call.get("function", {}).get("arguments") or "{}"
                args = _safe_json_args(raw_args)
                result = _run_tool(tools, name, args)
                steps.append({"tool": name, "arguments": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "name": name,
                    "content": json.dumps(result, default=str),
                })
        else:
            content = reply.get("content") or ""
            action = _parse_json_action(content)
            if action is None:
                # Model answered in prose instead of protocol JSON — treat as final.
                if content.strip():
                    return _done(content.strip(), steps, "json", reply["_model"])
                break  # empty, unparseable reply — force a final answer below
            if action.get("action") == "final":
                return _done(str(action.get("answer", "")).strip(), steps, "json", reply["_model"])
            name = str(action.get("action", ""))
            args = action.get("arguments") or {}
            result = _run_tool(tools, name, args if isinstance(args, dict) else {})
            steps.append({"tool": name, "arguments": args, "result": result})
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Tool result for {name}:\n{json.dumps(result, default=str)}",
            })

    # Ran out of steps, or the model stalled — force a plain-language final answer with no
    # further tools. In JSON mode the model may still wrap it in the protocol, so unwrap it.
    nudge = (
        "Now answer the user's question directly, in plain, friendly language, based on the "
        "tool results so far. Do not call any more tools."
    )
    if not use_tools:
        nudge += ' Reply with ONLY {"action": "final", "answer": "<your answer>"}.'
    final = _chat(models, messages + [{"role": "user", "content": nudge}], None)
    answer = (final.get("content") or "").strip()
    if not use_tools and answer:
        parsed = _parse_json_action(answer)
        if parsed is not None:
            answer = str(parsed.get("answer", "")).strip() if parsed.get("action") == "final" else ""
    return _done(answer, steps, "tools" if use_tools else "json", final["_model"])


def _run_tool(tools: Mapping[str, Tool], name: str, args: Mapping[str, Any]) -> Any:
    """Dispatch one tool call, returning its result or a structured error the model sees."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(tools)}."}
    try:
        return tool.run(args)
    except Exception as exc:  # tool bugs must not crash the loop — report them to the model
        logger.exception("Tool %s failed", name)
        return {"error": f"Tool '{name}' failed: {exc}"}


def _done(answer: str, steps: list[dict[str, Any]], mode: str, model: str) -> dict[str, Any]:
    return {
        "answer": answer or "I couldn't produce an answer this time.",
        "steps": steps,
        "mode": mode,
        "model": model,
    }


# --- Transport --------------------------------------------------------------


def _chat(
    models: Sequence[_Model],
    messages: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Call the chat endpoint across configured models until one succeeds.

    Returns a dict with ``content`` (str|None), ``tool_calls`` (list|None), and the private
    ``_model`` that answered. Raises :class:`_ToolsUnsupported` if a provider rejects the
    ``tools`` parameter (so the caller can retry in JSON mode), or :class:`AgentAPIError`
    if every model fails.
    """
    last_error: Exception | None = None
    for model in models:
        payload: dict[str, Any] = {
            "model": model.model,
            "temperature": 0.2,
            "messages": list(messages),
        }
        if tool_schemas:
            payload["tools"] = list(tool_schemas)
            payload["tool_choice"] = "auto"
        try:
            body = _post_chat(model.base_url, model.api_key, payload)
        except _ToolsUnsupported:
            raise  # bubble up immediately so run_agent switches to JSON mode
        except AgentAPIError as exc:
            last_error = exc
            logger.warning("Agent model %s failed: %s", model.model, exc)
            continue
        try:
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_error = AgentAPIError("The chat provider returned an unexpected response.")
            logger.warning("Agent model %s: malformed response: %s", model.model, exc)
            continue
        return {
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls"),
            "_model": model.model,
        }

    raise last_error or AgentAPIError("No agent model could be reached.")


def _post_chat(base_url: str, api_key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """POST to ``/chat/completions`` and return the parsed JSON body.

    Isolated so tests can stub it. Maps a "tools not supported" 400 to
    :class:`_ToolsUnsupported`, and other failures to :class:`AgentAPIError`.
    """
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise AgentAPIError(f"Could not reach the chat service: {exc}") from exc

    if response.status_code == 400 and "tool" in _safe_body(response).lower():
        raise _ToolsUnsupported()
    if response.status_code in (401, 403):
        logger.warning("Agent auth error (%s): %s", response.status_code, _safe_body(response))
        raise AgentAPIError("The chat service rejected the API key. Check LLM_API_KEY.")
    if not response.ok:
        logger.warning("Agent API %s: %s", response.status_code, _safe_body(response))
        raise AgentAPIError(f"The chat service returned an error (HTTP {response.status_code}).")
    try:
        return response.json()
    except ValueError as exc:
        raise AgentAPIError("The chat service returned invalid JSON.") from exc


# --- Config -----------------------------------------------------------------


@dataclass(frozen=True)
class _Model:
    api_key: str
    base_url: str
    model: str


def _load_models() -> list[_Model]:
    """Resolve the ordered agent models from LLM_* env, falling back to VISION_*.

    The primary model's key/base_url/model each fall back to the matching ``VISION_*``
    variable, then to the module defaults, so a user who only set ``VISION_API_KEY`` gets
    a working agent for free. Optional ``LLM_MODEL_2`` / ``LLM_MODEL_3`` add model
    fallbacks on the same key/endpoint. Attempts with no key are dropped; duplicates
    collapse. The key is never logged.
    """
    key = _env("LLM_API_KEY") or _env("VISION_API_KEY")
    base = (_env("LLM_BASE_URL") or _env("VISION_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    primary_model = _env("LLM_MODEL") or _env("VISION_MODEL") or DEFAULT_MODEL

    if not key:
        return []

    models: list[_Model] = []
    seen: set[tuple[str, str, str]] = set()
    model_ids = [primary_model]
    for index in range(2, _MAX_MODELS + 1):
        extra = _env(f"LLM_MODEL_{index}")
        if extra:
            model_ids.append(extra)

    for model_id in model_ids:
        triple = (key, base, model_id)
        if not model_id or triple in seen:
            continue
        seen.add(triple)
        models.append(_Model(api_key=key, base_url=base, model=model_id))
    return models


def _env(name: str) -> str:
    """Read an environment variable at call time, stripped ('' if unset)."""
    return os.getenv(name, "").strip()


# --- Small helpers ----------------------------------------------------------


def _parse_json_action(content: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model reply, or None if there isn't one.

    Tolerates code fences and leading/trailing prose, since free models are inconsistent.
    """
    text = content.strip()
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Find the first balanced-looking object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) and "action" in obj else None
    except ValueError:
        return None


def _safe_json_args(raw: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` JSON string, tolerating empties/garbage."""
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _as_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    """Coerce to an int clamped to [lo, hi], falling back to ``default``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _safe_body(response: requests.Response) -> str:
    """Return a short, log-safe snippet of a response body (never raises)."""
    try:
        return response.text[:500]
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"
