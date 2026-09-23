"""Tests for the tool-using Chef agent.

Fully offline: the chat transport (``agent._post_chat``) is stubbed with scripted replies,
and tools are simple fakes, so the ReAct controller, both execution modes, and the
auto-fallback are exercised without any network or API key.
"""

from __future__ import annotations

import json

import pytest

import agent

# --- Test doubles -----------------------------------------------------------


def _make_tools(calls_log):
    """A tiny tool set that records invocations into ``calls_log``."""

    def echo(args):
        calls_log.append(("echo", dict(args)))
        return {"echoed": args.get("value", None)}

    return {
        "echo": agent.Tool(
            name="echo",
            description="Echo the given value.",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            run=echo,
        )
    }


class _ScriptedTransport:
    """Stub for ``agent._post_chat`` that returns queued assistant messages in order."""

    def __init__(self, messages):
        self._queue = list(messages)
        self.calls = []

    def __call__(self, base_url, api_key, payload):
        self.calls.append(payload)
        message = self._queue.pop(0)
        if isinstance(message, Exception):
            raise message
        return {"choices": [{"message": message}]}


def _tool_call(name, arguments):
    return {
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


# --- configured() -----------------------------------------------------------


def test_configured_true_with_vision_key(monkeypatch):
    # conftest already sets VISION_API_KEY; agent falls back to it.
    monkeypatch.setenv("VISION_API_KEY", "test-key")
    assert agent.configured() is True


def test_configured_false_without_any_key(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    assert agent.configured() is False


def test_run_agent_raises_without_key(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(agent.AgentConfigError):
        agent.run_agent("hi", tools={})


# --- Native tool-calling mode -----------------------------------------------


def test_tools_mode_runs_tool_then_answers(monkeypatch):
    calls_log = []
    tools = _make_tools(calls_log)
    transport = _ScriptedTransport([
        _tool_call("echo", {"value": "paneer"}),
        {"content": "You said paneer.", "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("say paneer", tools=tools, mode="tools")

    assert result["answer"] == "You said paneer."
    assert result["mode"] == "tools"
    assert result["steps"] == [{"tool": "echo", "arguments": {"value": "paneer"}, "result": {"echoed": "paneer"}}]
    assert calls_log == [("echo", {"value": "paneer"})]
    # The second call carried the tool result back to the model.
    assert any(m.get("role") == "tool" for m in transport.calls[1]["messages"])


def test_tools_mode_recovers_from_empty_stall(monkeypatch):
    # Some free models call a tool, then return an empty message with no tool_calls. The
    # loop must not give up — it forces one plain-language final answer.
    calls_log = []
    tools = _make_tools(calls_log)
    transport = _ScriptedTransport([
        _tool_call("echo", {"value": "paneer"}),
        {"content": None, "tool_calls": None},          # stall: empty, tool-less reply
        {"content": "Use the paneer tonight.", "tool_calls": None},  # forced final answer
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("help", tools=tools, mode="tools")
    assert result["answer"] == "Use the paneer tonight."
    assert calls_log == [("echo", {"value": "paneer"})]
    # The forced-final call carried no tools schema.
    assert "tools" not in transport.calls[-1]


def test_json_mode_recovers_and_unwraps_forced_final(monkeypatch):
    tools = _make_tools([])
    transport = _ScriptedTransport([
        {"content": '{"action": "echo", "arguments": {"value": "x"}}', "tool_calls": None},
        {"content": "", "tool_calls": None},            # stall: empty reply
        {"content": '{"action": "final", "answer": "Wrapped answer."}', "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("go", tools=tools, mode="json")
    assert result["answer"] == "Wrapped answer."  # protocol JSON unwrapped to plain text


def test_unknown_tool_is_reported_not_crashed(monkeypatch):
    tools = _make_tools([])
    transport = _ScriptedTransport([
        _tool_call("does_not_exist", {}),
        {"content": "ok", "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("go", tools=tools, mode="tools")
    assert "error" in result["steps"][0]["result"]


# --- JSON-action fallback mode ----------------------------------------------


def test_json_mode_parses_action_then_final(monkeypatch):
    calls_log = []
    tools = _make_tools(calls_log)
    transport = _ScriptedTransport([
        {"content": '{"action": "echo", "arguments": {"value": "milk"}}', "tool_calls": None},
        {"content": '{"action": "final", "answer": "Milk it is."}', "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("use milk", tools=tools, mode="json")

    assert result["answer"] == "Milk it is."
    assert result["mode"] == "json"
    assert calls_log == [("echo", {"value": "milk"})]


def test_json_mode_tolerates_fenced_json(monkeypatch):
    tools = _make_tools([])
    transport = _ScriptedTransport([
        {"content": "Sure!\n```json\n{\"action\": \"final\", \"answer\": \"Done\"}\n```", "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("hi", tools=tools, mode="json")
    assert result["answer"] == "Done"


def test_json_mode_prose_reply_becomes_final(monkeypatch):
    tools = _make_tools([])
    transport = _ScriptedTransport([
        {"content": "Just eat the leftovers.", "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("advice?", tools=tools, mode="json")
    assert result["answer"] == "Just eat the leftovers."


# --- auto mode falls back on unsupported tools ------------------------------


def test_auto_mode_falls_back_to_json(monkeypatch):
    tools = _make_tools([])
    transport = _ScriptedTransport([
        agent._ToolsUnsupported(),  # first (tools) call: provider rejects tools
        {"content": '{"action": "final", "answer": "Fallback worked"}', "tool_calls": None},
    ])
    monkeypatch.setattr(agent, "_post_chat", transport)

    result = agent.run_agent("hello", tools=tools, mode="auto")
    assert result["answer"] == "Fallback worked"
    assert result["mode"] == "json"


# --- Default tools wired to (fake) storage ----------------------------------


def test_build_default_tools_plan_meals(monkeypatch):
    scan = {"items": [{"name": "paneer"}, {"name": "tomato"}, {"name": "onion"}]}
    tools = agent.build_default_tools(
        get_latest_scan=lambda: scan,
        list_perishables=lambda: [],
        list_nutrition=lambda: [],
    )
    assert set(tools) == {"get_inventory", "get_recommendations", "plan_meals"}

    inv = tools["get_inventory"].run({})
    assert inv["item_count"] == 3

    plan = tools["plan_meals"].run({"days": 1, "meals_per_day": 2})
    assert plan["solver"] in ("ilp", "greedy")
    assert "waste_avoided_pct" in plan["metrics"]
