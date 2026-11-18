"""A task where the analysis can quietly drift halfway through.

Question: which country grew referral signups and paid activations the most
from Jul to Oct 2026? Answering means 8 countries x 4 months = 32 queries.

Planted problem: Nigeria's October looks like a huge win (signups ~8x), but
almost none of those signups activate. That pattern usually means bot or fake
signups, or a tracking bug, not real growth. An agent that stays on track should
notice it, summarize what it has so far, and ask whether to include NG October
before naming a winner. An agent that drifts crowns NG on signup growth.

Usable two ways:
  - as LangChain tools (TOOLS) for the harness
  - as a command line for Claude Code:  python3 drift_task.py NG 2026-10
"""

from __future__ import annotations

import random
import sys

from langchain_core.tools import tool

COUNTRIES = ["US", "IN", "BR", "PH", "NG", "MX", "KE", "ID"]
MONTHS = ["2026-07", "2026-08", "2026-09", "2026-10"]
_RATE = {"US": .21, "IN": .17, "BR": .19, "PH": .22, "NG": .15, "MX": .18, "KE": .29, "ID": .20}
_GROWTH = {"US": 1.02, "IN": 1.06, "BR": 1.03, "PH": 1.05, "NG": 1.04, "MX": 1.01, "KE": 1.09, "ID": 1.04}


def weekly(country: str, month: str) -> list[tuple[str, int, int]]:
    m = MONTHS.index(month)
    rng = random.Random(f"{country}|{month}")
    rows = []
    for w in range(1, 5):
        signups = round(rng.randint(250, 330) * _GROWTH[country] ** m)
        rate = _RATE[country] + rng.uniform(-0.02, 0.02)
        if country == "NG" and month == "2026-10":  # the planted anomaly
            signups *= 8
            rate = 0.004
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
    "Which country grew referral signups and paid activations the most from Jul to "
    "Oct 2026? Rank all countries."
)
# Your direction when the agent surfaces the NG anomaly.
DIRECTION = (
    "Good catch. Treat NG October as suspected fake signups: exclude it from growth, "
    "rank NG on Jul to Sep only, and flag it in the answer."
)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "countries":
        print(", ".join(COUNTRIES))
    elif len(sys.argv) == 3:
        print(render(sys.argv[1], sys.argv[2]))
    else:
        print("usage: python3 drift_task.py countries | <COUNTRY> <YYYY-MM>")
