"""Offline tests for the two modes: quick (just runs) and steer (the full harness)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from example_tools import TOOLS
from scripted_model import SEEN, TOOLS_SEEN, ScriptedModel, call, used
from steerable import QUICK_PROMPT, SYSTEM_PROMPT, Budget, SteerInbox, build_agent


def run(agent, payload, cfg, on_tools=None):
    events, pending = [], None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["updates", "custom"]):
        if mode == "custom":
            events.append(chunk)
        elif "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
        elif "tools" in chunk and on_tools:
            on_tools()
    return events, pending


def system_text(messages):
    return str(messages[0].content)


# --- quick: no ledger, no check_in, no self-review; it just runs -------------

SEEN.clear(); TOOLS_SEEN.clear()
inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=[
    call("query_signups", {"month": "2026-07"}, 1),  # no ledger first: runs straight away
    call("query_signups", {"month": "2026-08"}, 2),
    call("query_signups", {"month": "2026-09"}, 3),
    used(call("query_signups", {"month": "2026-10"}, 4), input=6000),
    AIMessage("Referral grew. Assumed: September missing, left as a gap."),
]), TOOLS, inbox, mode="quick", review_every=2, budget=Budget(run=5000), checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "quick"}}
sent = []
events, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg,
                on_tools=lambda: sent or sent.append(inbox.put("referral only")))
msgs = agent.get_state(cfg).values["messages"]
results = [m for m in msgs if isinstance(m, ToolMessage)]
assert len(results) == 4 and not any(m.content.startswith("Blocked") for m in results), "nothing blocked"
assert not {"write_ledger", "update_ledger", "check_in"} & set(TOOLS_SEEN[0]), TOOLS_SEEN[0]
assert "query_signups" in TOOLS_SEEN[0]
assert QUICK_PROMPT.splitlines()[0] in system_text(SEEN[0]) and "write_ledger" not in system_text(SEEN[0])
assert not any("Self-review checkpoint" in str(m.content) for m in msgs), "no self-review prompts"
print("PASS quick mode: no ledger or check-in tools, nothing blocked, no self-review prompts")

assert inbox.items[0].status == "delivered" and any("referral only" in str(m.content) for m in msgs)
assert [e["step"] for e in events if e["type"] == "step_summary"] == [1, 2, 3, 4], "step panels still show"
assert p and p["type"] == "budget", "budgets still pause"
assert agent.get_state(cfg).values["harness_mode"] == "quick"
print("PASS quick mode keeps steering, step panels, budgets, and records its mode for --thread resume")

# --- steer: the full harness ---------------------------------------------------

SEEN.clear(); TOOLS_SEEN.clear()
agent = build_agent(ScriptedModel(responses=[call("query_signups", {"month": "2026-07"}, 1), AIMessage("x")]),
                    TOOLS, mode="steer", checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "steer"}}
run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
first = next(m for m in agent.get_state(cfg).values["messages"] if isinstance(m, ToolMessage))
assert first.content.startswith("Blocked"), "steer mode still requires the ledger first"
assert {"write_ledger", "update_ledger", "check_in"} <= set(TOOLS_SEEN[0])
assert SYSTEM_PROMPT.splitlines()[0] in system_text(SEEN[0])
print("PASS steer mode: ledger tools present and required, full instructions")

print("ALL-MODE-TESTS-OK")
