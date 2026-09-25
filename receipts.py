"""Receipt import — turn a pasted grocery receipt into structured fridge items.

A shopping trip is the *other* moment the fridge changes (the first being cooking). Rather
than make the user photograph the fridge again after every trip, this module reads the text
of a grocery receipt — pasted from a digital/email receipt, or OCR'd from a photo — and
extracts the items they bought, with quantities and prices, so the Chef can fold them back
into what's on hand.

It is deliberately **framework-agnostic** and **offline**: no Flask, no network, no LLM. It
is a deterministic text parser, so it is fast, free, private, and fully testable. The single
public entry point, :func:`parse_receipt`, returns a JSON-friendly preview the UI shows for
review before anything is written — the actual fridge write happens in
:func:`kitchen.import_receipt` only after the user confirms.

Real receipts are noisy — store headers, addresses, tax lines, payment tenders, loyalty
points, barcodes. The parser filters those out (see :data:`_META_PATTERNS`), pulls a price
and quantity off each remaining line, cleans the item name, and — using
:func:`recommender.is_known_ingredient` — flags whether each line looks like a real grocery
so the UI can pre-select the confident ones and let the shopper decide on the rest.
"""

from __future__ import annotations

import re
from typing import Any

import recommender

# --- Tunables ---------------------------------------------------------------

MAX_ITEMS = 80          # most line-items we'll return from one receipt
MAX_NAME = 80           # longest cleaned item name we keep
_MIN_ALPHA = 2          # a real item name has at least this many letters

# --- Line filters -----------------------------------------------------------

# Lines whose text contains any of these (case-insensitive) are receipt chrome, not items:
# totals, taxes, payment tenders, store/contact boilerplate, and column headers. Matched as
# whole words / substrings against the lowercased line.
_META_WORDS = (
    "subtotal", "sub total", "total", "grand total", "net total", "net amount",
    "amount due", "amount paid", "balance", "change due", "change", "tender", "tendered",
    "cash", "card", "credit", "debit", "visa", "mastercard", "master card", "maestro",
    "rupay", "amex", "upi", "paytm", "phonepe", "gpay", "wallet", "approved", "auth code",
    "tax", "cgst", "sgst", "igst", "gst", "vat", "hst", "pst", "service charge",
    "round off", "roundoff", "rounding", "discount", "savings", "you saved", "mrp",
    "qty", "quantity", "rate", "hsn", "sku", "barcode", "description", "particulars",
    "invoice", "bill no", "bill number", "receipt", "order no", "order id", "token no",
    "transaction", "ref no", "reference", "terminal", "batch", "trace",
    "thank", "thanks", "welcome", "visit again", "have a", "customer", "cashier",
    "counter", "operator", "store", "branch", "outlet", "phone", "tel:", "mobile",
    "gstin", "pan no", "fssai", "cin", "www.", "http", ".com", "email", "address",
    "terms", "return policy", "no exchange", "points", "loyalty", "member", "membership",
    "items sold", "no of items", "number of items", "items:", "date:", "time:",
)
_META_PATTERNS = tuple(re.compile(re.escape(w), re.IGNORECASE) for w in _META_WORDS)

# A line that is only digits, punctuation and spaces (separators, timestamps, barcodes).
_NON_ITEM_LINE = re.compile(r"^[\s\d\W]*$")
# A date-like line (2026-09-25, 25/09/26, 09.25.2026, etc.).
_DATE_LINE = re.compile(r"\b\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}\b")

# --- Money / quantity / unit extraction -------------------------------------

# A money token: a currency-prefixed number, or a bare number with exactly two decimals
# (the near-universal "price" shape on a receipt). Matches "₹27", "Rs.27.00", "$3.49",
# "27.00". A bare integer is intentionally NOT money here so it can be read as a quantity.
_MONEY = re.compile(r"(?:₹|rs\.?|inr|\$|usd)\s*\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*\.\d{2}\b", re.IGNORECASE)
# A trailing bare number, used only as a fallback price when no decimal/currency price exists.
_TRAILING_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[a-z]?\s*$", re.IGNORECASE)
# Explicit quantity markers: "2 x", "2x", "2 @", "2 ×" (leading), or "x 2" / "× 2" (trailing).
_QTY_LEADING = re.compile(r"^\s*(\d{1,3})\s*[x×@]\b", re.IGNORECASE)
_QTY_INLINE = re.compile(r"\b(\d{1,3})\s*[x×@]\s*(?=[₹$\d])", re.IGNORECASE)
_QTY_TRAILING = re.compile(r"[x×]\s*(\d{1,3})\s*$", re.IGNORECASE)
# A leading bare integer ("2 Milk") — a quantity only when a separate price was found.
_LEADING_INT = re.compile(r"^\s*(\d{1,3})\s+(?=[a-z])", re.IGNORECASE)

# Size/measure tokens ("500ml", "1 kg", "2%", "6 pack") — stripped from the name before
# normalization so it lands on the food ("amul milk 500ml" → milk) rather than the size.
_UNIT_TOKEN = re.compile(
    r"\b\d+(?:\.\d+)?\s?"
    r"(?:ml|l|ltr|litre|liter|g|gm|gms|gr|grm|kg|kgs|mg|oz|lb|lbs|ct|pk|pcs|pc|"
    r"pack|packs|dozen|doz|gal|kilo|kilos|bunch|tin|can|cans|btl|bottle|bottles)\b",
    re.IGNORECASE,
)
_PERCENT_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_LONG_DIGITS = re.compile(r"\b\d{5,}\b")           # barcodes / SKUs
_CURRENCY_SYMBOL = re.compile(r"(?:₹|rs\.?|inr|\$|usd)", re.IGNORECASE)
_STRAY_PUNCT = re.compile(r"[^\w%&+.\- ]+")


def parse_receipt(text: str, *, max_items: int = MAX_ITEMS) -> dict[str, Any]:
    """Parse pasted receipt ``text`` into a reviewable list of grocery items.

    Args:
        text: The raw receipt text (item lines, possibly with headers/totals/etc.).
        max_items: Cap on the number of line-items returned.

    Returns:
        A JSON-friendly dict::

            {
              "items": [
                {"name": str,       # cleaned, title-cased ("Amul Milk")
                 "token": str,      # canonical ingredient token (for merge/dedup)
                 "qty": int,        # >= 1
                 "price": float|None,
                 "known": bool,     # recognised as a real grocery
                 "raw": str},       # the original receipt line
                ...
              ],
              "count": int,         # number of items
              "known_count": int,   # of those, how many we recognised
              "total": float,       # sum of captured prices
              "currency": str,      # "₹" or "$"
              "lines_seen": int,    # non-blank lines examined
              "skipped": int,       # non-item lines filtered out
              "has_items": bool,
              "headline": str,      # plain-language summary
            }
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    currency = _detect_currency(text)
    parsed: list[dict[str, Any]] = []
    skipped = 0

    for line in lines:
        entry = _parse_line(line)
        if entry is None:
            skipped += 1
            continue
        parsed.append(entry)

    merged = _merge(parsed)[:max_items]
    total = round(sum(e["price"] for e in merged if e["price"]), 2)
    known_count = sum(1 for e in merged if e["known"])

    return {
        "items": merged,
        "count": len(merged),
        "known_count": known_count,
        "total": total,
        "currency": currency,
        "lines_seen": len(lines),
        "skipped": skipped,
        "has_items": bool(merged),
        "headline": _headline(merged, total, currency, known_count),
    }


# --- Internals --------------------------------------------------------------


def _parse_line(line: str) -> dict[str, Any] | None:
    """Turn one receipt line into an item entry, or ``None`` if it isn't an item."""
    if _is_meta(line):
        return None

    price = _extract_price(line)
    qty, working = _extract_qty(line, has_price=price is not None)
    name = _clean_name(working)

    if _alpha_count(name) < _MIN_ALPHA:
        return None  # no real name left after stripping numbers/units → not an item

    known = recommender.is_known_ingredient(name)
    # A grocery line-item carries a price; anything with neither a price nor a recognised
    # food name is almost certainly store/address boilerplate ("BIG BAZAAR", "123 MG Road")
    # that slipped past the keyword filters, so we drop it rather than pollute the fridge.
    if price is None and not known:
        return None

    token = recommender.normalize_ingredient(name) or name.lower()
    return {
        "name": _titlecase(name),
        "token": token,
        "qty": qty,
        "price": price,
        "known": known,
        "raw": line,
    }


def _is_meta(line: str) -> bool:
    """True for receipt chrome (headers, totals, tax, payment, barcodes, dates)."""
    if _NON_ITEM_LINE.match(line):
        return True
    if _DATE_LINE.search(line) and _alpha_count(line) < 4:
        return True  # a bare date/time line, but keep "Milk 2026 farms" style names
    return any(pat.search(line) for pat in _META_PATTERNS)


def _extract_price(line: str) -> float | None:
    """The line's price: the last currency/decimal money token, else a trailing number."""
    money = _MONEY.findall(line)
    if money:
        return _to_float(money[-1])
    trailing = _TRAILING_NUMBER.search(line)
    # Only treat a trailing bare number as a price when there's a name in front of it.
    if trailing and _alpha_count(line[: trailing.start()]) >= _MIN_ALPHA:
        return _to_float(trailing.group(1))
    return None


def _extract_qty(line: str, *, has_price: bool) -> tuple[int, str]:
    """Return ``(qty, line_without_qty_marker)``.

    Recognises explicit markers ("2 x", "2 @", "x2") anywhere, and a leading bare integer
    ("2 Milk") only when the line also carried a price (so a price isn't misread as a count).
    """
    for pattern in (_QTY_LEADING, _QTY_INLINE, _QTY_TRAILING):
        match = pattern.search(line)
        if match:
            qty = _as_int(match.group(1))
            return max(1, qty), line[: match.start()] + " " + line[match.end():]
    if has_price:
        match = _LEADING_INT.search(line)
        if match:
            return max(1, _as_int(match.group(1))), line[match.end():]
    return 1, line


def _clean_name(text: str) -> str:
    """Strip money, currency, units, percentages, barcodes and stray punctuation."""
    cleaned = _MONEY.sub(" ", text)
    cleaned = _CURRENCY_SYMBOL.sub(" ", cleaned)
    cleaned = _UNIT_TOKEN.sub(" ", cleaned)
    cleaned = _PERCENT_TOKEN.sub(" ", cleaned)
    cleaned = _LONG_DIGITS.sub(" ", cleaned)
    cleaned = _TRAILING_NUMBER.sub(" ", cleaned)   # drop a leftover trailing price/number
    cleaned = _STRAY_PUNCT.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop any leftover standalone number words at the edges (e.g. a bare "2" or "27").
    words = [w for w in cleaned.split() if not w.replace(".", "").isdigit()]
    return " ".join(words).strip(" .-")


def _merge(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine repeated items (same token): sum quantities and prices, keep first name."""
    order: list[str] = []
    by_token: dict[str, dict[str, Any]] = {}
    for entry in entries:
        token = entry["token"]
        if token in by_token:
            existing = by_token[token]
            existing["qty"] += entry["qty"]
            if entry["price"] is not None:
                existing["price"] = round((existing["price"] or 0.0) + entry["price"], 2)
        else:
            by_token[token] = dict(entry)
            order.append(token)
    return [by_token[t] for t in order]


def _headline(items: list[dict[str, Any]], total: float, currency: str, known: int) -> str:
    """A one-line, plain-language summary of what we found."""
    if not items:
        return "I couldn't spot grocery items in that — paste the item lines from your receipt."
    count = len(items)
    noun = "item" if count == 1 else "items"
    money = f" worth about {currency}{_money_str(total)}" if total else ""
    if known < count:
        unknown = count - known
        tail = f" ({unknown} I wasn't sure about — uncheck any that aren't food)."
    else:
        tail = " — review and add them to your fridge."
    return f"Found {count} {noun}{money}{tail}"


# --- Small value helpers ----------------------------------------------------


def _detect_currency(text: str) -> str:
    """Pick a display currency from the symbols present; default to ₹ (the app default)."""
    blob = str(text or "")
    if "₹" in blob or re.search(r"\brs\b|\binr\b", blob, re.IGNORECASE):
        return "₹"
    if "$" in blob or re.search(r"\busd\b", blob, re.IGNORECASE):
        return "$"
    return "₹"


def _to_float(token: str) -> float | None:
    """Parse a money token ("₹1,299.00", "27") into a float, or ``None`` if unparseable."""
    digits = re.sub(r"[^\d.]", "", str(token))
    if not digits or digits == ".":
        return None
    try:
        return round(float(digits), 2)
    except ValueError:
        return None


def _alpha_count(text: str) -> int:
    """How many alphabetic characters ``text`` contains."""
    return sum(1 for ch in text if ch.isalpha())


def _titlecase(name: str) -> str:
    """Title-case a cleaned item name, leaving all-caps acronyms readable."""
    words = []
    for word in name.split():
        words.append(word if (word.isupper() and len(word) <= 3) else word.capitalize())
    return " ".join(words)[:MAX_NAME]


def _as_int(value: Any) -> int:
    """Coerce to int, defaulting to 1."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _money_str(amount: float) -> str:
    """Format a rupee/dollar amount without a trailing ``.0`` for whole numbers."""
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}"
