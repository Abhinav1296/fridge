"""Offline Chef: a rule-based fallback that drives the same tools as the LLM agent.

When no chat model is configured — or the provider is down — the Chef tab shouldn't go
dark. This module recognises the common requests with plain regular expressions
("I'm vegetarian", "I'm allergic to peanuts", "add milk to my shopping list", "I cooked
palak paneer", "I threw away the spinach", "plan 3 days", "what's expiring?", ...), calls
the very same :class:`agent.Tool` objects the LLM would, and writes a templated reply.

It obeys the same policy as the LLM agent, because it *uses the same tools*: preferences
and the shopping list are updated directly, while anything that changes the fridge only
becomes a pending action the user must confirm. It returns the same result shape as
:func:`agent.run_agent` (with ``mode="offline"`` and ``model="rules"``), so the web layer
and UI treat both identically. Framework-agnostic and fully deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import agent

_AND = re.compile(r"\s*(?:,|\band\b|&|\+)\s*", re.IGNORECASE)
_FILLER = re.compile(r"^(?:the|some|a|an|my|our|all the|all|of the)\s+", re.IGNORECASE)

# --- Preference statements (side intents: several can appear in one message) ----------

_DIET = re.compile(
    r"\b(?:i\s*am|i'm|im|we\s*are|we're|am|i\s+eat|we\s+eat|i\s+follow|we\s+follow|go)\s+"
    r"(?:a\s+|an\s+|strictly\s+|pure\s+|only\s+)?"
    r"(vegetarian|vegan|eggetarian|jain|veg|non[- ]?veg(?:etarian)?)\b",
    re.IGNORECASE,
)
_NO_MEAT = re.compile(r"\b(?:i|we)\s+(?:don't|do not|dont|never)\s+eat\s+meat\b", re.IGNORECASE)
_ALLERGY = re.compile(r"\ballergic\s+to\s+([a-z ,&+]+?)(?=[.!?;]|\s+(?:so|but|what|can|and\s+(?:i|we))\b|$)",
                      re.IGNORECASE)
_ALLERGY_NOUN = re.compile(r"\b(?:i\s+have|i've got|we\s+have)\s+(?:a|an)\s+([a-z ]+?)\s+allergy\b",
                           re.IGNORECASE)
_LACTOSE = re.compile(r"\blactose[- ]intolerant\b", re.IGNORECASE)
_DISLIKE = re.compile(
    r"\b(?:i|we)\s+(?:don't|do not|dont|really don't)\s+(?:like|enjoy|eat)\s+([a-z ,&+]+?)"
    r"(?=[.!?;]|\s+(?:so|but|what|can)\b|$)|\b(?:i|we)\s+(?:hate|dislike|can't stand)\s+"
    r"([a-z ,&+]+?)(?=[.!?;]|\s+(?:so|but|what|can)\b|$)",
    re.IGNORECASE,
)
_HOUSEHOLD = re.compile(
    r"\b(?:cook(?:ing)?\s+for|family\s+of|household\s+of|there\s+are|we\s+are)\s+(\d{1,2})\b"
    r"|\b(\d{1,2})\s+(?:people|of\s+us|persons|members)\b",
    re.IGNORECASE,
)

# --- Primary intents (the first that matches wins) ------------------------------------

_SHOP_ADD = re.compile(
    r"\b(?:add|put)\s+(.+?)\s+(?:to|on(?:to)?|in)\s+(?:my\s+|the\s+|our\s+)?(?:shopping|grocery)\s*list\b",
    re.IGNORECASE,
)
_SHOP_NEED = re.compile(r"\b(?:we|i)\s+need\s+to\s+buy\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)
_SHOP_SHOW = re.compile(r"\b(?:shopping|grocery)\s*list\b|\bwhat\s+(?:do|should)\s+(?:i|we)\s+(?:need\s+to\s+)?buy\b",
                        re.IGNORECASE)
_WASTED = re.compile(
    r"\b(?:threw|throw|thrown|tossed|binned|chucked)\s+(?:away|out)?\s*(.+?)(?:\s+(?:away|out))?(?:[.!?]|$)"
    r"|\b(.+?)\s+(?:went\s+bad|went\s+off|spoiled|spoilt|rotted|got\s+mouldy|got\s+moldy)\b",
    re.IGNORECASE,
)
_USED = re.compile(r"\b(?:finished|used\s+up|ate\s+all\s+(?:of\s+)?)\s*(?:the\s+)?(.+?)(?:[.!?]|$)",
                   re.IGNORECASE)
_COOKED = re.compile(
    r"\b(?:i|we)\s+(?:just\s+|have\s+|already\s+)?(?:cooked|made|prepared|ate|had)\s+(.+?)"
    r"(?:\s+(?:for|at|today|tonight|yesterday|last\s+night)\b.*)?(?:[.!?]|$)",
    re.IGNORECASE,
)
_TRACK = re.compile(
    r"\b(?:track\s+|remind\s+me\s+about\s+)?(?:the\s+|my\s+)?([a-z][a-z ]{1,40}?)\s+"
    r"(?:expires?|expiring|goes\s+(?:bad|off)|is\s+good\s+(?:till|until)|use[- ]by(?:\s+is)?)\s+"
    r"(?:(?:in\s+(\d{1,2})\s+days?)|(tomorrow)|(today)|(?:on\s+)?(\d{4}-\d{2}-\d{2}))",
    re.IGNORECASE,
)
_PLAN = re.compile(r"\bplan\b|\bmeal\s*plan\b|\bnext\s+\d+\s+days\b|\bfor\s+the\s+week\b", re.IGNORECASE)
_EXPIRING = re.compile(r"expir|going\s+(?:bad|off)|use\s+soon|use\s+up|about\s+to\s+go|at\s+risk",
                       re.IGNORECASE)
_WASTE_REPORT = re.compile(r"\bwaste\b|\bwasting\b|\bwasted\b|\bmoney\b|\bsav(?:e|ed|ing)\b", re.IGNORECASE)
_PREFS = re.compile(r"\bwhat\s+do\s+you\s+(?:know|remember)\b|\bmy\s+preferences\b", re.IGNORECASE)
_SEARCH = re.compile(
    r"\brecipes?\s+(?:for|with|using)\b|\bhow\s+(?:do\s+i|to|can\s+i)\s+(?:make|cook)\b"
    r"|\bsomething\s+with\b|\bdish(?:es)?\s+with\b|\b(?:find|search\s+for)\s+(?:me\s+)?(?:a|an|some)\b"
    r"|\b(?:quick|easy|spicy|healthy|light|simple|creamy|crispy|high[- ]protein|low[- ]carb)\b"
    r".*\b(?:dinner|lunch|breakfast|meal|snack|dish|recipe|curry)\b",
    re.IGNORECASE,
)
_INVENTORY = re.compile(
    r"\bwhat(?:'s|\s+is)\s+in\s+(?:my|the)\s+fridge\b|\binventory\b|\bwhat\s+do\s+(?:i|we)\s+have\b",
    re.IGNORECASE,
)


def run_offline(
    message: str,
    *,
    tools: Mapping[str, agent.Tool],
    on_step: agent.StepCallback | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Answer ``message`` with rules + tools. Same result shape as :func:`agent.run_agent`."""
    text = " ".join(str(message or "").split())
    state = agent._RunState()

    def call(tool: str, /, **args: Any) -> Any:  # positional-only: tools take a ``name`` arg
        if tool not in tools:
            return {"error": f"'{tool}' isn't available."}
        return agent._dispatch(tools, tool, args, state, on_step)

    parts: list[str] = []
    remembered = _remember(text, call)
    if remembered:
        parts.append(remembered)

    reply = _primary(text, call, tools, skip_default=bool(remembered))
    if reply:
        parts.append(reply)

    answer = " ".join(parts) or "Tell me what you'd like — for example “what can I cook?”."
    result = {
        "answer": answer,
        "steps": state.steps,
        "mode": "offline",
        "model": "rules",
        "actions": list(state.actions.values()),
        "changed": sorted(state.changed),
    }
    if note:
        result["note"] = note
    return result


# --- Preference statements ------------------------------------------------------------


def _remember(text: str, call) -> str:
    saved: list[str] = []
    problems: list[str] = []

    def save(key: str, value: Any, label: str) -> None:
        res = call("remember_preference", key=key, value=value)
        if isinstance(res, Mapping) and res.get("saved"):
            saved.append(label)
        elif isinstance(res, Mapping) and res.get("error"):
            problems.append(str(res["error"]))

    diet = _DIET.search(text)
    if diet:
        word = diet.group(1).lower().replace(" ", "-")
        save("diet", "any" if word.startswith("non") else word,
             "you eat anything" if word.startswith("non") else f"you're {_DIET_NAMES.get(word, word)}")
    elif _NO_MEAT.search(text):
        save("diet", "vegetarian", "you're vegetarian")

    allergies = [*_split_all(_ALLERGY.findall(text)), *_split_all(_ALLERGY_NOUN.findall(text))]
    if _LACTOSE.search(text):
        allergies.append("dairy")
    if allergies:
        save("allergies", ", ".join(allergies), f"you're allergic to {_join(allergies)}")

    dislikes = _split_all([a or b for a, b in _DISLIKE.findall(text)])
    if dislikes:
        save("dislikes", ", ".join(dislikes), f"you don't like {_join(dislikes)}")

    size = _HOUSEHOLD.search(text)
    if size:
        count = size.group(1) or size.group(2)
        save("household_size", int(count), f"you cook for {count}")

    if not saved and not problems:
        return ""
    out = f"Got it — I'll remember that {_join(saved)}." if saved else ""
    if problems:
        out = (out + " " if out else "") + "I couldn't save one thing: " + problems[0]
    return out


# --- Primary intents ------------------------------------------------------------------


def _primary(text: str, call, tools: Mapping[str, agent.Tool], *, skip_default: bool) -> str:
    match = _SHOP_ADD.search(text) or _SHOP_NEED.search(text)
    if match:
        items = _split(match.group(1))
        if not items:
            return "What should I add to the shopping list?"
        res = call("add_to_shopping_list", items=[{"name": i} for i in items])
        if res.get("error"):
            return res["error"]
        added = res.get("added") or []
        out = f"Added {_join(added)} to your shopping list." if added else ""
        if res.get("already_on_list"):
            out += f" {res['already_on_list']} {'was' if res['already_on_list'] == 1 else 'were'} already on it."
        if res.get("refused_for_allergy"):
            out += f" I left off {_join(res['refused_for_allergy'])} because of an allergy."
        return out.strip()

    match = _TRACK.search(text)
    if match:
        name, days, tomorrow, today, iso = match.groups()
        args: dict[str, Any] = {"name": _clean_name(name)}
        if iso:
            args["use_by"] = iso
        else:
            args["days_from_now"] = int(days) if days else (1 if tomorrow else 0)
        return _proposal(call("propose_track_expiry", **args))

    match = _WASTED.search(text)
    if match:
        name = _clean_name(match.group(1) or match.group(2) or "")
        if name:
            return _proposal(call("propose_log_item", name=name, outcome="wasted"))

    match = _USED.search(text)
    if match and _clean_name(match.group(1)):
        return _proposal(call("propose_log_item", name=_clean_name(match.group(1)), outcome="used"))

    match = _COOKED.search(text)
    if match and _clean_name(match.group(1)):
        dish = _clean_name(match.group(1))
        dish, _, used = dish.partition(" with ")
        res = call("propose_cook_meal", recipe=dish)
        if res.get("error") and "don't know a recipe" in res["error"] and _split(used):
            # Not a catalog dish, but they told us what went in: "I cooked X with A and B".
            res = call("propose_cook_meal", recipe=dish, ingredients=_split(used))
        if res.get("error") and "don't know a recipe" in res["error"]:
            # "I ate the yogurt" isn't a recipe — maybe it's a tracked item they finished.
            logged = call("propose_log_item", name=dish, outcome="used")
            if not logged.get("error"):
                return _proposal(logged)
            return (f"I couldn't find a recipe or a fridge item called '{dish}'. If it's a dish, "
                    "tell me what went into it (e.g. “I cooked X with rice and dal”).")
        return _proposal(res)

    if _SHOP_SHOW.search(text):
        return _say_shopping(call("get_shopping_list"))
    if _PREFS.search(text):
        res = call("get_preferences")
        return f"Here's what I remember: {res.get('summary')}." if not res.get("error") else res["error"]
    if _PLAN.search(text):
        days = _first_int(r"(\d+)\s*days?", text) or (7 if re.search(r"\bweek\b", text, re.I) else 3)
        meals = _first_int(r"(\d+)\s*meals?", text) or 2
        return _say_plan(call("plan_meals", days=days, meals_per_day=meals))
    if _EXPIRING.search(text):
        return _say_expiring(call("get_inventory"), call("get_recommendations"))
    if _WASTE_REPORT.search(text) and "get_waste_report" in tools:
        return _say_waste(call("get_waste_report"))
    if _SEARCH.search(text) and "search_recipes" in tools:
        return _say_search(call("search_recipes", query=text, k=4))
    if _INVENTORY.search(text):
        return _say_inventory(call("get_inventory"))
    if skip_default and not re.search(r"\?|\bcook\b|\bmake\b|\bsuggest\b|\beat\b", text, re.I):
        return ""
    return _say_recommendations(call("get_recommendations"))


# --- Reply templates ------------------------------------------------------------------


def _proposal(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    action = res.get("action") or {}
    out = f"I've prepared this for you: {action.get('summary')}. Tap Confirm to apply it."
    if res.get("warnings"):
        out += f" Heads up: {_join(res['warnings'])}."
    if res.get("not_in_fridge"):
        out += f" (Not in the fridge, so not deducted: {_join(res['not_in_fridge'])}.)"
    return out


def _say_inventory(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    items = [str(i.get("name")) for i in res.get("items") or [] if i.get("name")]
    if not items:
        return "Your fridge looks empty — scan it and I'll take it from there."
    out = f"You have {len(items)} item{'s' if len(items) != 1 else ''}: {_join(items[:12])}"
    out += f" and {len(items) - 12} more." if len(items) > 12 else "."
    tracked = [t for t in res.get("tracked_use_by") or [] if t.get("days_left") is not None]
    if tracked:
        out += " Use-by dates: " + "; ".join(
            f"{t['name']} {_days_phrase(t['days_left'])}" for t in tracked[:5]
        ) + "."
    return out


def _say_recommendations(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    recipes = res.get("recipes") or []
    if not recipes:
        return "I couldn't find a dish that fits what's in the fridge. Try scanning it again."
    ready = [r["title"] for r in recipes if r.get("can_make")]
    almost = [r for r in recipes if not r.get("can_make")]
    parts = []
    if ready:
        parts.append(f"You can cook right now: {_join(ready[:3])}.")
    if almost:
        r = almost[0]
        parts.append(f"{'Also close' if ready else 'Closest'}: {r['title']} (needs {_join(r.get('missing') or [])}).")
    urgent = [u["name"] for u in res.get("use_soon") or [] if u.get("severity") == "soon"]
    if urgent:
        parts.append(f"Use soon: {_join(urgent[:4])}.")
    return " ".join(parts)


def _say_expiring(inv: Mapping[str, Any], rec: Mapping[str, Any]) -> str:
    rows = rec.get("use_soon") or [] if not rec.get("error") else []
    if not rows:
        return "Nothing is about to go off. Nice!"
    out = "Needs attention: " + "; ".join(f"{u['name']} ({u['reason']})" for u in rows[:6]) + "."
    uses = [r for r in rec.get("recipes") or [] if r.get("uses_expiring")]
    if uses:
        out += f" {uses[0]['title']} would use {_join(uses[0]['uses_expiring'])}."
    return out


def _say_plan(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    rows = res.get("plan") or []
    if not rows:
        return "I couldn't build a plan from what's in the fridge — try scanning it first."
    lines = [f"{i}. {r['title']}" + (f" (uses {_join(r['uses_expiring'])} before it goes off)"
                                      if r.get("uses_expiring") else "")
             for i, r in enumerate(rows, start=1)]
    out = "Here's your plan: " + " ".join(lines)
    shopping = [s.get("item", s) if isinstance(s, Mapping) else s for s in res.get("shopping_list") or []]
    out += f" To buy: {_join([str(s) for s in shopping[:8]])}." if shopping else " No shopping needed!"
    return out


def _say_waste(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    if not res.get("has_data"):
        return "No food has been logged as eaten or thrown away yet, so there's no waste report."
    out = str(res.get("headline") or "")
    top = [t.get("name") for t in res.get("top_wasted") or [] if t.get("name")]
    if top:
        out += f" Most wasted: {_join(top[:3])}."
    return out.strip()


def _say_search(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    results = res.get("results") or []
    if not results:
        return "I couldn't find a matching recipe."
    bits = []
    for r in results[:3]:
        need = "you have everything" if r.get("can_make") else f"needs {_join(r.get('missing')[:3])}"
        bits.append(f"{r['title']} ({need})")
    return "Try: " + "; ".join(bits) + "."


def _say_shopping(res: Mapping[str, Any]) -> str:
    if res.get("error"):
        return str(res["error"])
    rows = res.get("to_buy") or []
    if not rows:
        return "Your shopping list is empty."
    return "On your list: " + _join([r["name"] + (f" ({r['qty']})" if r.get("qty") else "") for r in rows[:15]]) + "."


# --- Helpers ----------------------------------------------------------------------------

_DIET_NAMES = {"veg": "vegetarian"}


def _split(text: str) -> list[str]:
    return [n for n in (_clean_name(p) for p in _AND.split(text or "")) if n]


def _split_all(chunks: Sequence[str]) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        for name in _split(chunk):
            if name.lower() not in (o.lower() for o in out):
                out.append(name)
    return out


def _clean_name(text: str) -> str:
    name = " ".join(str(text or "").split()).strip(" .,!?;:'\"")
    while True:
        stripped = _FILLER.sub("", name)
        if stripped == name:
            break
        name = stripped
    return name[:60]


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _days_phrase(days: int) -> str:
    if days < 0:
        return f"{-days} day{'s' if days != -1 else ''} overdue"
    if days == 0:
        return "today"
    return f"in {days} day{'s' if days != 1 else ''}"


def _join(names: Sequence[str]) -> str:
    return agent._join([str(n) for n in names if str(n)])
