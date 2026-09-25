"""Proactive watcher — Chef speaks first.

The chat agent only acts when asked. The watcher looks at the fridge on its own — after
every scan, on a timer, after a confirmed kitchen action, or when the user taps
*Check now* — and files a short **briefing**: what's about to go off, what can be cooked
with it right now, and what to buy. Every suggestion that would change something (throw
X out, mark a meal cooked, add to the shopping list) is filed as a **pending action**:
nothing happens until the user presses Confirm.

Deterministic by design: the facts, the suggestions and a template headline all come
from the recommender (the self-trained embeddings + the household's saved rules). An LLM,
when configured and ``AGENT_BRIEFING_LLM`` isn't ``"0"``, may only reword the headline —
the lines and actions underneath are always the template's. Unchanged facts don't produce
a new briefing (a hash of the facts is compared with the last one).

Framework-agnostic: no Flask imports. The web layer decides *when* to call
:func:`generate_briefing` and pushes the realtime update.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import agent
import preferences
import recommender
import restock
import storage

logger = logging.getLogger(__name__)

SOURCE = "briefing"                  # action-queue source for everything the watcher files
TRIGGERS = frozenset({"scan", "timer", "manual", "action"})
DEFAULT_WATCH_MINUTES = 60
MAX_LINES_SOON = 5                   # "use soon" rows listed in one briefing
MAX_RESTOCK = 3                      # staples nudged for restock in one briefing
MAX_HEADLINE = 200

_DISCARD = ("overdue", "spoiled")
_lock = threading.Lock()             # one briefing at a time (scan + timer can race)


def watch_minutes() -> int:
    """Minutes between timer checks (``AGENT_WATCH_MINUTES``; ``0`` disables the timer)."""
    raw = os.getenv("AGENT_WATCH_MINUTES", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_WATCH_MINUTES
    except ValueError:
        value = DEFAULT_WATCH_MINUTES
    return max(0, value)


def generate_briefing(
    trigger: str, *, now: datetime | None = None, force: bool = False
) -> dict[str, Any] | None:
    """Look at the fridge and file a briefing; return it, or ``None`` if there's nothing new.

    Args:
        trigger: What woke the watcher: ``scan``, ``timer``, ``action`` or ``manual``.
        now: Reference time (UTC) for expiry math; defaults to now.
        force: File a briefing even when the facts haven't changed or nothing is urgent
            (the manual *Check now* button always gets an answer).

    When there's nothing to report (and not forced), any still-visible older briefing is
    dismissed and its buttons expired, so the banner never shows stale advice.
    """
    trigger = trigger if trigger in TRIGGERS else "manual"
    now = now or datetime.now(UTC)
    with _lock:
        memory = storage.get_memory()
        inventory = storage.get_current_inventory(normalize=recommender.normalize_ingredient)
        items = list((inventory or {}).get("items") or [])
        perishables = storage.list_perishables()
        rec = recommender.recommend(
            items, perishables, now=now, recipe_filter=preferences.build_filter(memory)
        )
        shopping_rows = storage.list_shopping(include_done=False)
        on_list = {str(row.get("item")) for row in shopping_rows}
        low = _restock_facts(items, perishables, shopping_rows, now)
        facts = _facts(rec, memory, on_list)
        facts["restock"] = low
        digest = _digest(facts)
        previous = storage.latest_briefing(include_dismissed=True)

        if not force and previous and previous.get("facts_hash") == digest:
            return None
        if not force and not _worth_saying(facts):
            if previous and not previous.get("dismissed"):
                storage.dismiss_briefing(int(previous["id"]))
                storage.expire_actions(source=SOURCE)
            return None

        storage.expire_actions(source=SOURCE)  # yesterday's buttons go stale with the facts
        actions = _file_actions(facts, inventory, perishables, memory, now)
        template = _headline(facts)
        polished = _polish(template, _lines(facts))
        body = {
            "lines": _lines(facts),
            "actions": actions,
            "facts": facts,
            "worded_by": "llm" if polished else "template",
        }
        return storage.save_briefing(trigger, polished or template, body, digest)


# --- Facts ------------------------------------------------------------------------


def _facts(
    rec: Mapping[str, Any], memory: Mapping[str, Any], on_list: set[str]
) -> dict[str, Any]:
    """Reduce a recommender result to the few facts a briefing is about."""
    use_soon = [
        {k: row.get(k) for k in ("name", "token", "severity", "reason", "days_left")}
        for row in rec.get("use_soon") or []
    ]
    discard = [row for row in use_soon if row["severity"] in _DISCARD]
    soon = [row for row in use_soon if row["severity"] == "soon"][:MAX_LINES_SOON]

    cook = None
    buy = None
    blocked = preferences.excluded_tokens(memory)
    for recipe in rec.get("recipes") or []:
        if not recipe.get("uses_expiring"):
            continue
        if cook is None and recipe.get("can_make"):
            cook = {"id": recipe.get("id"), "title": recipe.get("title"),
                    "uses_expiring": list(recipe["uses_expiring"])}
        elif buy is None and not recipe.get("can_make") and 1 <= len(recipe.get("missing") or []) <= 3:
            missing = [m for m in recipe["missing"]
                       if recommender.normalize_ingredient(m) not in blocked]
            if len(missing) != len(recipe["missing"]):
                continue  # would need an excluded ingredient — don't suggest buying for it
            buy = {"id": recipe.get("id"), "title": recipe.get("title"),
                   "uses_expiring": list(recipe["uses_expiring"]),
                   "missing": [m for m in missing
                               if recommender.normalize_ingredient(m) not in on_list],
                   "already_on_list": [m for m in missing
                                       if recommender.normalize_ingredient(m) in on_list]}
    return {"discard": discard, "soon": soon, "cook": cook, "buy": buy}


def _restock_facts(
    items: Sequence[Mapping[str, Any]],
    perishables: Sequence[Mapping[str, Any]],
    shopping_rows: Sequence[Mapping[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """The few staples the briefing should nudge a restock for (most pressing first).

    Read from the long-term waste log, so this reflects what the household actually goes
    through — not just the last scan. Defensive: any failure here degrades to "no restock
    nudge" rather than breaking the whole briefing.
    """
    try:
        events = storage.list_waste_events()
        result = restock.suggest_restock(
            events, items, perishables=perishables, on_list=shopping_rows,
            now=now, top_n=MAX_RESTOCK,
        )
    except Exception:  # noqa: BLE001 — a restock hiccup must never sink the briefing
        logger.exception("Restock suggestion failed; skipping the nudge")
        return []
    return list(result.get("items") or [])


def _worth_saying(facts: Mapping[str, Any]) -> bool:
    # An "out" staple is worth a briefing on its own; "low" only rides along with other news.
    out = [r for r in facts.get("restock") or [] if r.get("status") == "out"]
    return bool(facts["discard"] or facts["soon"] or facts["cook"] or out)


def _digest(facts: Mapping[str, Any]) -> str:
    """Stable hash of what the briefing would say (names/days, not wording)."""
    key = {
        "discard": [(r["token"], r["severity"], r["days_left"]) for r in facts["discard"]],
        "soon": [(r["token"], r["days_left"]) for r in facts["soon"]],
        "cook": (facts["cook"] or {}).get("id"),
        "buy": [(facts["buy"] or {}).get("id"), (facts["buy"] or {}).get("missing")],
        "restock": [(r["token"], r["status"]) for r in facts.get("restock") or []],
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:32]


# --- Actions ------------------------------------------------------------------------


def _file_actions(
    facts: Mapping[str, Any],
    inventory: Mapping[str, Any] | None,
    perishables: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    """File the briefing's confirm-gated suggestions; return them (id, tool, summary)."""
    filed: list[dict[str, Any]] = []

    def keep(action: Mapping[str, Any] | None) -> None:
        if action and action.get("id") is not None:
            filed.append({k: action.get(k) for k in ("id", "tool", "summary", "status")})

    if facts["discard"]:
        names = [str(r["name"]) for r in facts["discard"]]
        keep(storage.create_action(
            "discard_items",
            {"items": [{"name": n, "qty": 1} for n in names]},
            f"Throw out {agent._join(names)} and log {'it' if len(names) == 1 else 'them'} "
            "as wasted",
            source=SOURCE,
        ))

    if facts["cook"]:
        # Reuse the chat agent's own proposal logic: same allergy block, same "deduct only
        # what's actually in the fridge" rule, same summary wording.
        tools = agent.build_default_tools(
            get_latest_scan=lambda: inventory,
            list_perishables=lambda: perishables,
            get_memory=lambda: memory,
            propose=lambda tool, args, summary: storage.create_action(
                tool, args, summary, source=SOURCE),
            now=lambda: now,
        )
        result = tools["propose_cook_meal"].run({"recipe": facts["cook"]["id"]})
        keep((result or {}).get("action"))

    buy = facts["buy"]
    if buy and buy["missing"]:
        keep(storage.create_action(
            "add_shopping",
            {"items": [{"name": m, "reason": f"for {buy['title']}"} for m in buy["missing"]]},
            f"Add {agent._join(buy['missing'])} to the shopping list (for {buy['title']})",
            source=SOURCE,
        ))

    # Restock nudge: only staples you've fully run out of get a confirm-gated add — "low"
    # ones ride along in the wording without pre-filing a button.
    out = [r for r in facts.get("restock") or [] if r.get("status") == "out"]
    if out:
        names = [str(r["name"]) for r in out]
        keep(storage.create_action(
            "add_shopping",
            {"items": [{"name": n, "reason": "running low"} for n in names]},
            f"Add {agent._join(names)} to the shopping list (you're out and use "
            f"{'it' if len(names) == 1 else 'them'} often)",
            source=SOURCE,
        ))
    return filed


# --- Wording ------------------------------------------------------------------------


def _headline(facts: Mapping[str, Any]) -> str:
    """One-sentence summary, most pressing thing first."""
    discard = [str(r["name"]) for r in facts["discard"]]
    soon = [str(r["name"]) for r in facts["soon"]]
    cook = facts["cook"]
    if discard:
        verb = "is" if len(discard) == 1 else "are"
        text = f"{agent._join(discard)} {verb} past {'its' if len(discard) == 1 else 'their'} best"
        if cook:
            return f"{text} — and you can make {cook['title']} right now."
        return f"{text} — time to check and clear {'it' if len(discard) == 1 else 'them'} out."
    if soon and cook:
        return f"Use {agent._join(soon[:3])} soon — you can make {cook['title']} right now."
    if soon:
        return f"Use {agent._join(soon[:3])} soon."
    if cook:
        return f"You can make {cook['title']} right now."
    out = [str(r["name"]) for r in (facts.get("restock") or []) if r.get("status") == "out"]
    if out:
        verb = "one" if len(out) == 1 else "some"
        return f"You're out of {agent._join(out[:3])} — worth grabbing {verb} on your next shop."
    return "Nothing urgent in your fridge right now."


def _lines(facts: Mapping[str, Any]) -> list[str]:
    """The briefing's detail lines (always template-worded)."""
    lines = [f"{r['name']}: {r['reason']}." for r in facts["discard"]]
    lines += [f"{r['name']}: {r['reason']}." for r in facts["soon"]]
    if facts["cook"]:
        c = facts["cook"]
        lines.append(f"Cook now: {c['title']} (uses {agent._join(c['uses_expiring'])}).")
    buy = facts["buy"]
    if buy and buy["missing"]:
        lines.append(f"Buy {agent._join(buy['missing'])} to make {buy['title']} "
                     f"(uses {agent._join(buy['uses_expiring'])}).")
    elif buy and buy["already_on_list"]:
        lines.append(f"Everything for {buy['title']} is already on your shopping list.")
    for r in facts.get("restock") or []:
        verb = "out of" if r.get("status") == "out" else "running low on"
        lines.append(f"You're {verb} {r['name']} — {r['reason']}")
    if not lines:
        lines.append("Nothing is close to its use-by date. Nice work.")
    return lines


_POLISH_SYSTEM = (
    "You reword fridge alerts for a home-cooking app. Reply with ONE friendly sentence of at "
    "most 25 words and nothing else. Use only the facts given: never add a food, dish, "
    "number or date that isn't listed."
)


def _polish(template: str, lines: Sequence[str]) -> str | None:
    """Optionally let the LLM reword the headline; ``None`` keeps the template."""
    if os.getenv("AGENT_BRIEFING_LLM", "1").strip() == "0" or not agent.configured():
        return None
    prompt = "Facts:\n" + "\n".join(f"- {line}" for line in lines) + f"\n\nDraft: {template}"
    try:
        text = agent.complete(prompt, system=_POLISH_SYSTEM)
    except agent.AgentError as exc:
        logger.info("Briefing headline left as template (%s)", exc)
        return None
    text = " ".join(text.replace("**", "").split()).strip().strip('"“”')  # plain-text banner
    if not text or len(text) > MAX_HEADLINE:
        return None
    return text
