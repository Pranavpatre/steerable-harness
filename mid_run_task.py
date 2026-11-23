"""A task whose key assumption only shows up halfway, and only you can resolve it.

Question: which country grew referral signups and paid activations the most from
Jul to Oct 2026? Nothing in the question is ambiguous, so asking first finds
nothing to ask.

Planted gap: India's Sep and Oct numbers jump about 1.8x and activations rise with
them, so the data looks like real growth and a careful agent has no reason to flag
it. Only you know the jump is a one-off partner import that should not count.
You can only notice it once you see India's Sep numbers come back. With India's
import left out, Kenya grew the most; with it in, India wins.

All data is synthetic and deterministic (seeded).
"""

from __future__ import annotations

import random

from langchain_core.tools import tool

COUNTRIES = ["US", "IN", "BR", "PH", "NG", "MX", "KE", "ID"]
MONTHS = ["2026-07", "2026-08", "2026-09", "2026-10"]
_RATE = {"US": .21, "IN": .17, "BR": .19, "PH": .22, "NG": .15, "MX": .18, "KE": .29, "ID": .20}
_GROWTH = {"US": 1.02, "IN": 1.06, "BR": 1.03, "PH": 1.05, "NG": 1.04, "MX": 1.01, "KE": 1.09, "ID": 1.04}
IMPORT_MONTHS = {"2026-09", "2026-10"}


def weekly(country: str, month: str) -> list[tuple[str, int, int]]:
    m = MONTHS.index(month)
    rng = random.Random(f"mid|{country}|{month}")
    rows = []
    for w in range(1, 5):
        signups = round(rng.randint(250, 330) * _GROWTH[country] ** m)
        if country == "IN" and month in IMPORT_MONTHS:  # the planted gap: looks like real growth
            signups = round(signups * 1.8)
        rate = _RATE[country] + rng.uniform(-0.02, 0.02)
        rows.append((f"{month} wk{w}", signups, round(signups * rate)))
    return rows


def render(country: str, month: str) -> str:
    if country not in COUNTRIES or month not in MONTHS:
        return f"No data for {country} {month}."
    body = "\n".join(f"{wk} | signups {s} | paid_activated {a}" for wk, s, a in weekly(country, month))
    return f"{country} {month}\nweek | signups | paid_activated\n{body}"


@tool
def list_countries() -> str:
    """List the countries that have referral signup data."""
    return ", ".join(COUNTRIES)


@tool
def query_referral_signups(country: str, month: str) -> str:
    """Weekly referral signups and paid activations for one country and month.

    Args:
        country: Two-letter country code from list_countries.
        month: Month formatted YYYY-MM, between 2026-07 and 2026-10.
    """
    return render(country, month)


TOOLS = [list_countries, query_referral_signups]

QUESTION = (
    "Which country grew referral signups and paid activations the most from Jul to Oct 2026? "
    "Rank all countries. Start your answer with a line 'Winner: <country code>'."
)
# What you say once you see India's Sep numbers and recognise the import.
DIRECTION = (
    "Heads up: India's Sep and Oct 2026 numbers include a one-off partner import, not organic "
    "referral growth. Leave India's Sep and Oct out of growth, rank India on Jul to Aug only, "
    "and flag it in the answer."
)
# What you say if the agent asks before it has shown you any data: you don't know about the import yet.
NO_PREFERENCE = "No special definitions. Use the data as it is."
LEADER = "KE"


def is_gap_result(text: str) -> bool:
    """True when a tool result shows India's import months: the moment you can spot it."""
    return text.startswith("IN ") and any(m in text for m in IMPORT_MONTHS)


def true_leader() -> str:
    def growth(c: str, last: str) -> float:
        first = sum(s for _, s, _ in weekly(c, "2026-07"))
        end = sum(s for _, s, _ in weekly(c, last))
        return end / first
    return max(COUNTRIES, key=lambda c: growth(c, "2026-08" if c == "IN" else "2026-10"))
