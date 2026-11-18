"""Live check on a real model: does the agent stop at the data gaps instead of assuming?

Reads ANTHROPIC_API_KEY (or another provider's key) from .env. Set AGENT_MODEL to
try another model, e.g. AGENT_MODEL=deepseek:deepseek-chat.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from example_tools import TOOLS
from steerable import _print_check_in, _print_update, build_agent

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5")
QUESTION = (
    "Compare monthly referral vs paid_meta vs organic signups for Jul to Oct 2026, "
    "and the activation rate per channel. Which channel is best?"
)
ANSWER = (
    "Leave September as a gap, don't interpolate it. Treat organic activation as "
    "unknown and rank only referral vs paid_meta."
)

agent = build_agent(MODEL, TOOLS, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "live"}}
payload = {"messages": [HumanMessage(QUESTION)]}
check_ins = []

print(f"model: {MODEL}\nyou> {QUESTION}")
for _ in range(4):  # at most 3 check-ins, then stop
    pending = None
    for update in agent.stream(payload, cfg, stream_mode="updates"):
        if "__interrupt__" in update:
            pending = update["__interrupt__"][0].value
        else:
            _print_update(update)
    if pending is None:
        break
    check_ins.append(pending)
    _print_check_in(pending)
    print(f"you> {ANSWER}")
    payload = Command(resume=ANSWER)

print("\n" + "=" * 60)
print(f"RESULT: {len(check_ins)} check-in(s)")
for c in check_ins:
    print("  gaps flagged:", c["gaps_or_concerns"])
print("PASS: agent stopped to ask" if check_ins else "FAIL: agent never checked in")
