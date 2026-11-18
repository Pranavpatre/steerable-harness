"""Offline tests for Phase 2: assumption ledger, pre-tool pauses, restart-safe checkpoints.

The Phase 2 gate is the first test: a run pauses at the first checkpoint (the
ledger), the process exits, and a NEW process resumes it from the SQLite file.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from example_tools import TOOLS
from scripted_model import ScriptedModel, call
from steerable import SteerInbox, ToolGuardMiddleware, build_agent, current_ledger

LEDGER = {
    "scope": ["Jul to Oct 2026", "referral channel"],
    "inputs": ["query_signups per month"],
    "constraints": ["do not interpolate missing months"],
    "done_when": ["monthly table + trend"],
}


def run(agent, payload, cfg):
    events, pending = [], None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["updates", "custom"]):
        if mode == "custom":
            events.append(chunk)
        elif "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
    return events, pending


def state(agent, cfg):
    return agent.get_state(cfg).values


# --- GATE: pauses at the first checkpoint and resumes after a restart (A1, A7) ---

db = Path(tempfile.mkdtemp()) / "checkpoints.sqlite"
PROC = """
import sys
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from example_tools import TOOLS
from scripted_model import ScriptedModel, call
from steerable import build_agent, sqlite_checkpointer, current_ledger
LEDGER = {ledger!r}
phase, db = sys.argv[1], Path(sys.argv[2])
cfg = {{"configurable": {{"thread_id": "restart-gate"}}}}
if phase == "start":
    model = ScriptedModel(responses=[call("write_ledger", LEDGER, 1)])
    agent = build_agent(model, TOOLS, checkpointer=sqlite_checkpointer(db))
    out = agent.invoke({{"messages": [HumanMessage("Signups trend?")]}}, cfg)
    print("PAUSED", out["__interrupt__"][0].value["type"])
else:
    # A brand-new process: only the SQLite file carries the paused run.
    model = ScriptedModel(responses=[call("query_signups", {{"month": "2026-08"}}, 2), AIMessage("Aug: 4,390.")])
    agent = build_agent(model, TOOLS, checkpointer=sqlite_checkpointer(db))
    snap = agent.get_state(cfg)
    print("FOUND", snap.interrupts[0].value["type"])
    agent.invoke(Command(resume="ok"), cfg)
    v = agent.get_state(cfg).values
    print("CONFIRMED", current_ledger(v["ledger_ops"])[1])
    print("LAST", v["messages"][-1].content)
""".format(ledger=LEDGER)
here = Path(__file__).parent
a = subprocess.run([sys.executable, "-c", PROC, "start", str(db)], cwd=here, capture_output=True, text=True)
b = subprocess.run([sys.executable, "-c", PROC, "resume", str(db)], cwd=here, capture_output=True, text=True)
assert "PAUSED ledger" in a.stdout, a.stdout + a.stderr
assert "FOUND ledger" in b.stdout and "CONFIRMED True" in b.stdout and "LAST Aug: 4,390." in b.stdout, b.stdout + b.stderr
print("PASS GATE paused at the ledger checkpoint in one process, resumed and finished in another")

# --- A1: no tool runs before the ledger is confirmed ------------------------

agent = build_agent(ScriptedModel(responses=[
    call("query_signups", {"month": "2026-08"}, 1),  # jumps the gun
    call("write_ledger", LEDGER, 2),
    call("query_signups", {"month": "2026-08"}, 3),
    AIMessage("Done."),
]), TOOLS, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "a1"}}
_, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert p and p["type"] == "ledger" and p["ledger"]["constraints"] == LEDGER["constraints"], p
first = next(m for m in state(agent, cfg)["messages"] if isinstance(m, ToolMessage))
assert first.status == "error" and first.content.startswith("Blocked"), first
assert "4390" not in str(state(agent, cfg)["messages"]), "the early query must not have run"
run(agent, Command(resume=""), cfg)
results = [m for m in state(agent, cfg)["messages"] if isinstance(m, ToolMessage) and m.name == "query_signups"]
assert len(results) == 2 and "4390" in results[1].content, "after confirming, the query runs"
print("PASS a tool called before the ledger is blocked; the agent is re-prompted; it runs after you confirm")

# --- A2, A3: your edits and the agent's changes show up as ledger changes ---

inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=[
    call("write_ledger", LEDGER, 1),
    call("update_ledger", {"change": "revised", "field": "scope", "item": "Jul to Oct 2026",
                           "new_item": "Aug to Oct 2026", "reason": "you said skip July"}, 2),
    call("query_signups", {"month": "2026-08"}, 3),
    AIMessage("Done."),
]), TOOLS, inbox, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "a2"}}
run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
inbox.put("/ledger constraints: exclude bot signups")  # typed while it works
events, _ = run(agent, Command(resume="skip July"), cfg)
ledger, confirmed = current_ledger(state(agent, cfg)["ledger_ops"])
assert confirmed and ledger["scope"] == ["Aug to Oct 2026", "referral channel"], ledger
assert ledger["constraints"] == ["do not interpolate missing months", "exclude bot signups"], ledger
assert inbox.items[0].status == "delivered"
assert any("[Ledger edit from the user] constraints" in str(m.content) for m in state(agent, cfg)["messages"])
changes = {e["step"]: e["ledger_changes"] for e in events if e["type"] == "step_summary"}
assert any("~ scope: Jul to Oct 2026 → Aug to Oct 2026 (agent)" in c for c in changes[2]), changes
assert any("edit" in c for c in changes[1]), "your correction at the checkpoint is logged"
print("PASS your ledger edits land as steers; the agent's revisions show in the step panel:")
print("     step 2 ledger:", "; ".join(changes[2]))

# --- A5, A6: ask before a subagent spawn or a costly call; decline re-prompts --

guard = ToolGuardMiddleware(require_ledger=False)
agent = build_agent(ScriptedModel(responses=[
    call("task", {"description": "research every country", "subagent_type": "general-purpose"}, 1),
    AIMessage("OK, I'll do it inline instead."),
]), TOOLS, guard=guard, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "a5"}}
_, p = run(agent, {"messages": [HumanMessage("research")]}, cfg)
assert p and p["type"] == "approval" and p["calls"][0]["reasons"] == ["spawns a subagent"], p
run(agent, Command(resume="no, do it yourself"), cfg)
msgs = state(agent, cfg)["messages"]
declined = next(m for m in msgs if isinstance(m, ToolMessage))
assert declined.status == "error" and "declined" in declined.content
assert msgs[-1].content == "OK, I'll do it inline instead."
print("PASS a subagent spawn pauses for approval; declining feeds back to the model without running it")

guard = ToolGuardMiddleware(require_ledger=False, budget_tokens=100, budget_share=0.25)
big = {"file_path": "/big.md", "content": "x" * 400}
agent = build_agent(ScriptedModel(responses=[call("write_file", big, 1), AIMessage("Written.")]),
                    TOOLS, guard=guard, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "a5b"}}
_, p = run(agent, {"messages": [HumanMessage("write it")]}, cfg)
assert p and "over 25% of the 100-token budget" in p["calls"][0]["reasons"][0], p
assert p["calls"][0]["est_tokens"] > 25
run(agent, Command(resume=""), cfg)
assert "/big.md" in state(agent, cfg)["files"], "approved call runs"
print("PASS a call estimated over the budget share pauses with its estimate; approving runs it")

# --- one interrupt per node: approval and ledger block in the same batch -----

agent = build_agent(ScriptedModel(responses=[
    AIMessage(content="", tool_calls=[
        {"name": "write_ledger", "args": LEDGER, "id": "w1"},
        {"name": "query_signups", "args": {"month": "2026-08"}, "id": "q1"},
    ]),
    AIMessage("Done."),
]), TOOLS, checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "batch"}}
_, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert p["type"] == "ledger"
run(agent, Command(resume="ok"), cfg)
q = next(m for m in state(agent, cfg)["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "q1")
assert q.content.startswith("Blocked"), "a call in the same batch as write_ledger still waits for confirmation"
print("PASS a query batched with write_ledger is held back until the ledger is confirmed")

