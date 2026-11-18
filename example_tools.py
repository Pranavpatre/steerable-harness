"""Toy analytics tools with deliberate gaps, to exercise check_in.

Replace these with your real tools (Redash, SQL, APIs). The data below has a
missing month and an uninstrumented column, the kind of gap the agent should
stop and ask about instead of papering over.
"""

from langchain_core.tools import tool

_SIGNUPS = {
    "2026-07": {"referral": 4120, "paid_meta": 9800, "organic": 6100},
    "2026-08": {"referral": 4390, "paid_meta": 10250, "organic": 5900},
    # 2026-09 is missing on purpose
    "2026-10": {"referral": 5010, "paid_meta": 9900, "organic": 6400},
}
_ACTIVATION = {"referral": 0.41, "paid_meta": 0.22, "organic": None}  # organic not tracked


@tool
def list_tables() -> str:
    """List the tables available in the database."""
    return "signups_by_month_channel(month, channel, signups); activation_rate(channel, rate)"


@tool
def query_signups(month: str) -> str:
    """Return signups by channel for a month formatted YYYY-MM."""
    row = _SIGNUPS.get(month)
    return str(row) if row else f"No rows for {month}."


@tool
def query_activation_rate(channel: str) -> str:
    """Return the activation rate for a channel."""
    if channel not in _ACTIVATION:
        return f"Unknown channel {channel}."
    rate = _ACTIVATION[channel]
    return "NULL (not instrumented)" if rate is None else f"{rate:.0%}"


TOOLS = [list_tables, query_signups, query_activation_rate]
