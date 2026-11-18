"""A longer toy task where a silent assumption is easy to make and expensive to undo.

The question asks for "activation rate" by country. The database has three
activation definitions and no default, and the best country changes depending
on which one you use. The answer needs 8 countries x 4 months = 32 queries per
definition, so guessing wrong and redoing it costs real tokens.

All data is synthetic and deterministic (seeded), so every run sees the same numbers.
"""

from __future__ import annotations

import random

from langchain_core.tools import tool

COUNTRIES = ["US", "IN", "BR", "PH", "NG", "MX", "KE", "ID"]
MONTHS = ["2026-07", "2026-08", "2026-09", "2026-10"]
DEFINITIONS = {
    "first_task_7d": "Completed a first task within 7 days of signup.",
    "first_task_30d": "Completed a first task within 30 days of signup.",
    "paid_task_30d": "Completed a task that passed review and was paid, within 30 days of signup.",
}
# Base activation rates. The leader differs by definition: IN on first_task_7d,
# PH on first_task_30d, KE on paid_task_30d.
_BASE = {
    "first_task_7d":  {"US": .31, "IN": .44, "BR": .33, "PH": .40, "NG": .36, "MX": .30, "KE": .34, "ID": .35},
    "first_task_30d": {"US": .46, "IN": .55, "BR": .48, "PH": .59, "NG": .50, "MX": .44, "KE": .52, "ID": .49},
    "paid_task_30d":  {"US": .21, "IN": .17, "BR": .19, "PH": .22, "NG": .15, "MX": .18, "KE": .29, "ID": .20},
}

# What the user actually means. Only they know this.
INTENDED = "paid_task_30d"


def _weekly(country: str, month: str, definition: str) -> list[tuple[str, int, int]]:
    rng = random.Random(f"{country}|{month}|{definition}")
    sign_rng = random.Random(f"{country}|{month}")  # signups don't depend on definition
    rows = []
    for w in range(1, 5):
        signups = sign_rng.randint(180, 420)
        rate = max(0.01, _BASE[definition][country] + rng.uniform(-0.03, 0.03))
        rows.append((f"{month} wk{w}", signups, round(signups * rate)))
    return rows


@tool
def list_activation_definitions() -> str:
    """List the activation definitions available in the database."""
    lines = [f"- {k}: {v}" for k, v in DEFINITIONS.items()]
    return "Activation definitions (there is no default; pick one per query):\n" + "\n".join(lines)


@tool
def list_countries() -> str:
    """List the countries that have referral signup data."""
    return ", ".join(COUNTRIES)


@tool
def query_referral_activation(country: str, month: str, definition: str) -> str:
    """Weekly referral signups and activated users for one country and month.

    Args:
        country: Two-letter country code from list_countries.
        month: Month formatted YYYY-MM, between 2026-07 and 2026-10.
        definition: One of the names from list_activation_definitions.
    """
    if definition not in DEFINITIONS:
        return f"Unknown definition {definition!r}. Use list_activation_definitions."
    if country not in COUNTRIES or month not in MONTHS:
        return f"No data for {country} {month}."
    rows = _weekly(country, month, definition)
    body = "\n".join(f"{wk} | signups {s} | activated {a}" for wk, s, a in rows)
    return f"{country} {month} ({definition})\nweek | signups | activated\n{body}"


TOOLS = [list_activation_definitions, list_countries, query_referral_activation]

QUESTION = (
    "Using referral signups from Jul to Oct 2026, which country has the best activation "
    "rate? Give me the rate per country and say how confident you are in the ranking."
)
CLARIFICATION = (
    f"By activation I mean {INTENDED}: a paid task within 30 days. "
    "Use only that definition."
)


def true_leader(definition: str = INTENDED) -> str:
    """The correct answer, computed from the data, for checking the agent."""
    totals = {}
    for c in COUNTRIES:
        s = a = 0
        for m in MONTHS:
            for _, signups, act in _weekly(c, m, definition):
                s, a = s + signups, a + act
        totals[c] = a / s
    return max(totals, key=totals.get)
